"""Parser for a UE4 `StaticMaterials` array value blob
(`ArrayProperty<StructProperty<FStaticMaterial>>`).

This module implements ONLY the array-descriptor + element-walking logic that
``uasset.properties.read_properties`` cannot perform by itself: ``read_properties``
stores the whole ``StaticMaterials`` value as a raw ``bytes`` blob (see
``_read_property_old`` -> the ``ArrayProperty`` branch, ``reader.read_bytes(size)``)
because a generic property reader has no idea how to iterate a *struct* array's
tagged elements. We take that blob back apart here.

ON-DISK LAYOUT (determined from the UE4.27 engine source, see citations below)
-------------------------------------------------------------------------------

``raw`` is exactly the bytes that ``read_properties(...)['StaticMaterials']``
returned, i.e. the *value* of the outer ``ArrayProperty``. It is produced by
``FArrayProperty::SerializeItem`` and consists of, in order:

    [ count : int32 ]                          -- FArrayProperty::SerializeItem, PropertyArray.cpp:180
                                                 (``Slot.EnterArray(n)`` writes ``n``)
    [ inner struct PropertyTag descriptor ]     -- PropertyArray.cpp:205-211
                                                 (``UnderlyingArchive << MaybeInnerTag``)
    [ element 0 ]                              -- PropertyArray.cpp:326 (Inner->SerializeItem)
    [ element 1 ]
    ...
    [ element count-1 ]

The inner *descriptor* is a single OLD-format (UE4.27) ``FPropertyTag`` whose
``Name`` is the array field name (e.g. ``'StaticMaterials'``), ``Type`` is
``'StructProperty'``, ``Size`` is the TOTAL size of every element combined
(e.g. 984 == 3 * 328), ``StructName`` is ``'StaticMaterial'`` and ``StructGuid``
is 16 bytes. Its layout is defined by ``operator<<(FStructuredArchive::FSlot, FPropertyTag&)``
in PropertyTag.cpp:81-205:

    Name(FName:8) Type(FName:8) Size(int32:4) ArrayIndex(int32:4)
    [ if StructProperty: StructName(FName:8) + StructGuid(16) ]
    HasPropertyGuid(1 byte) [ + PropertyGuid(16) if set ]

=> descriptor is **49 bytes** for a struct with no property GUID.

EACH ELEMENT is itself a tagged property block, because
``UScriptStruct::SerializeItem`` (Class.cpp:2718) takes the
``SerializeTaggedProperties`` branch for ``FStaticMaterial``: the struct is NOT a
native serializer (no ``TStructOpsTypeTraits<FStaticMaterial>`` with
``WithSerializer`` exists in the engine tree) and these uncooked assets do not use
binary property serialization. ``SerializeTaggedProperties`` writes one
``FPropertyTag`` per reflected ``UPROPERTY`` field of ``FStaticMaterial``
(StaticMesh.h:444-457) followed by an ``'None'`` FName terminator:

    MaterialInterface  : ObjectProperty  (FStaticMaterial::MaterialInterface, StaticMesh.h:445)
                         value = FPackageIndex int32 (4 bytes)
    MaterialSlotName   : NameProperty    (StaticMesh.h:449)   value = FName (8 bytes)
    ImportedMaterialSlotName : NameProperty (StaticMesh.h:453, WITH_EDITORONLY_DATA)
                         value = FName (8 bytes)
    UVChannelData      : StructProperty  (StaticMesh.h:457, FMeshUVChannelInfo, Components.h:66)
                         value = Size bytes (176 for these meshes)
    None terminator    : FName 'None' (8 bytes)

(The flat ``operator<<(FArchive&, FStaticMaterial&)`` in StaticMesh.cpp:2723 is
the NATIVE/binary order and is NOT used on disk for tagged exports -- it is only
the in-memory field order, listed here for completeness.)

Verified empirically (SM_Ceiling_Scaffold_Quarter.uasset, raw len 1037):
    count @0 = 3
    descriptor @4..53 (49 bytes), Size=984
    element 0 @53, stride = 328 == 29 + 33 + 33 + (49 + 176) + 8
        MaterialInterface value @ elem+25  (e.g. byte 78)  -> FPackageIndex, NOT 4-aligned
        MaterialSlotName    value @ elem+54
        ImportedMaterialSlotName value @ elem+87 (byte 140 == 'Padding')
        UVChannelData       value @ elem+148 (176 bytes)
        None terminator     @ elem+320 (8 bytes)
    element i @ 53 + i*328 ; final reader position == raw length (no trailing bytes).
"""

from typing import List, Tuple, Optional

from .reader import BinaryReader


# ---------------------------------------------------------------------------
# OLD-format (UE4.27) FPropertyTag header reader
# ---------------------------------------------------------------------------

def _read_fname(reader: BinaryReader, name_map) -> str:
    """Read a serialized FName = (DisplayIndex int32, Number int32) and resolve it.

    This is the same canonical 8-byte FName read that ``read_properties`` performs
    internally (properties.py: name_idx + name_num); we read it via BinaryReader
    rather than re-deriving it from anything else.
    """
    idx = reader.read_int32()
    _number = reader.read_int32()  # FName.Number (suffix digit); unused for resolution
    if 0 <= idx < len(name_map):
        return name_map[idx]
    return f"#{idx}"


def _read_old_tag_header(reader: BinaryReader, name_map) -> Optional[dict]:
    """Read one OLD-format (UE4.27) ``FPropertyTag`` *header*.

    Source: ``operator<<(FStructuredArchive::FSlot, FPropertyTag&)`` --
    PropertyTag.cpp:81-205.

    Leaves ``reader`` positioned at the start of the property *value* (the ``Size``
    bytes). Returns ``None`` when the tag Name is ``'None'`` (the struct/array
    terminator -- PropertyTag.cpp:95-98 returns immediately after the Name).
    """
    # PropertyTag.cpp:94  Name (FName)
    name = _read_fname(reader, name_map)
    if name == "None":
        # PropertyTag.cpp:95-98  -- terminator, no further bytes follow.
        return None

    # PropertyTag.cpp:101  Type (FName)
    type_name = _read_fname(reader, name_map)
    # PropertyTag.cpp:113  Size (int32)
    size = reader.read_int32()
    # PropertyTag.cpp:114  ArrayIndex (int32)
    _array_index = reader.read_int32()

    struct_name = None
    if type_name == "StructProperty":
        # PropertyTag.cpp:122-135  StructName (FName) + StructGuid (16 bytes)
        struct_name = _read_fname(reader, name_map)
        reader.skip(16)  # StructGuid (VER_UE4_STRUCT_GUID_IN_PROPERTY_TAG)

    # PropertyTag.cpp:189-204  HasPropertyGuid (1 byte) [+ PropertyGuid (16)]
    has_guid = reader.read_uint8()
    if has_guid:
        reader.skip(16)

    return {"name": name, "type": type_name, "size": size, "struct": struct_name}


def _skip_value_for_unknown(reader: BinaryReader, tag: dict) -> None:
    """Advance past a property value we don't need to interpret (by Size)."""
    reader.skip(tag["size"])


def _read_element(reader: BinaryReader, name_map) -> dict:
    """Walk one ``FStaticMaterial`` element = tagged properties until ``'None'``.

    Implements the per-element ``SerializeTaggedProperties`` walk
    (Class.cpp:2775) for the four reflected fields of FStaticMaterial
    (StaticMesh.h:445/449/453/457). Only ``MaterialInterface`` (ObjectProperty ->
    FPackageIndex int32) and the two NameProperties are decoded; ``UVChannelData``
    (StructProperty) is skipped by its Size. Stops at the ``'None'`` terminator.
    """
    fields: dict = {}
    while True:
        tag = _read_old_tag_header(reader, name_map)
        if tag is None:
            return fields  # consumed the 'None' FName terminator

        name = tag["name"]
        typ = tag["type"]
        size = tag["size"]

        if typ == "ObjectProperty" and size >= 4:
            # FPackageIndex int32 (UMaterialInterface* ref) -- MaterialInterface.
            fields[name] = reader.read_int32()
            reader.skip(size - 4)
        elif typ == "NameProperty" and size >= 8:
            # FName value (idx int32 + number int32) -- MaterialSlotName /
            # ImportedMaterialSlotName.
            fields[name] = _read_fname(reader, name_map)
            reader.skip(size - 8)
        else:
            # StructProperty (UVChannelData) or anything else: skip Size bytes.
            _skip_value_for_unknown(reader, tag)


def _resolve_material_import(package_index, imports) -> Optional[str]:
    """Resolve an FPackageIndex to an object name.

    FPackageIndex convention: 0 == null; <0 == import table index ``-idx-1``;
    >0 == export table index ``idx-1``. Imports carry ``object_name`` (the
    material asset name, e.g. ``Mi_Beam_Thin``); export refs are rare for material
    slots and we don't have the export table here, so we tag them ``#expN``.
    """
    if package_index is None or not isinstance(package_index, int):
        return None
    if package_index < 0:
        imp = -package_index - 1
        if imports is not None and 0 <= imp < len(imports):
            obj = imports[imp]
            return getattr(obj, "object_name", None)
        return f"#imp{imp}"
    if package_index > 0:
        return f"#exp{package_index - 1}"
    return None  # package_index == 0 -> null material


def parse_static_materials_array(
    raw: bytes, name_map, imports
) -> Optional[List[Tuple[str, str]]]:
    """Parse a ``StaticMaterials`` ``ArrayProperty<StructProperty>`` VALUE blob.

    Parameters
    ----------
    raw : bytes
        The blob returned by ``read_properties(...)['StaticMaterials']``:
        ``int32 count`` + the inner struct PropertyTag descriptor + ``count``
        tagged elements.
    name_map : list[str]
        The package name table (``Package.name_map``), used to resolve FNames.
    imports : list
        The package import table (``Package.imports``); entries expose
        ``object_name``. Used to turn each slot's ``MaterialInterface``
        FPackageIndex into the material asset name.

    Returns
    -------
    list[tuple[str, str]] or None
        Ordered (one entry per material slot, in slot order) tuples of
        ``(ImportedMaterialSlotName, material_import_name)``. Returns ``None`` on
        any structural failure so callers can fall back safely.
    """
    try:
        reader = BinaryReader(raw)

        # [0:4] array element count  -- FArrayProperty::SerializeItem (PropertyArray.cpp:180)
        count = reader.read_int32()
        if count < 0 or count > 1_000_000:
            return None

        # [4:53] inner struct descriptor -- PropertyArray.cpp:205-211.
        # This is ONE StructProperty tag header; its "value" (Size bytes) IS the
        # concatenated elements, so we read its header and then walk the elements
        # directly -- we do NOT consume its Size here.
        descriptor = _read_old_tag_header(reader, name_map)
        if descriptor is None or descriptor.get("type") != "StructProperty":
            return None

        result: List[Tuple[str, str]] = []
        for _ in range(count):
            fields = _read_element(reader, name_map)
            slot_name = fields.get("ImportedMaterialSlotName")
            material = _resolve_material_import(fields.get("MaterialInterface"), imports)
            result.append((slot_name, material))
        return result
    except Exception:
        # Any malformed/truncated blob -> fail safe so callers fall back.
        return None
