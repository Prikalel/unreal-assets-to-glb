"""Property tag parser/skipper for UE5 .uasset files.

Handles both old format (UE4 style) and new format (UE5 >= PROPERTY_TAG_COMPLETE_TYPE_NAME).
"""
from .reader import BinaryReader
from .package import (
    UE5_PROPERTY_TAG_COMPLETE_TYPE_NAME,
    UE5_PROPERTY_TAG_EXTENSION,
)


# EPropertyTagFlags bits
TAG_HasArrayIndex = 0x01
TAG_HasPropertyGuid = 0x02
TAG_HasPropertyExtensions = 0x04
TAG_HasBinaryOrNativeSerialize = 0x08
TAG_BoolTrue = 0x10
TAG_SkippedSerialize = 0x20


def _read_property_type_name(r: BinaryReader, name_map):
    """Read FPropertyTypeName (tree of FName + InnerCount nodes)."""
    total_nodes = 1
    type_name = ""
    i = 0
    while i < total_nodes:
        idx = r.read_int32()
        _num = r.read_int32()  # FName number
        inner_count = r.read_int32()
        name = name_map[idx] if 0 <= idx < len(name_map) else f"#{idx}"
        if i == 0:
            type_name = name
        total_nodes += inner_count
        i += 1
    return type_name


def skip_properties(reader: BinaryReader, name_map, file_version_ue5: int) -> int:
    """Skip all property tags until "None" is encountered.

    Returns the number of properties skipped.
    The reader should be positioned at the start of the export data.
    """
    use_new_format = file_version_ue5 >= UE5_PROPERTY_TAG_COMPLETE_TYPE_NAME
    use_extensions = file_version_ue5 >= UE5_PROPERTY_TAG_EXTENSION

    count = 0
    while True:
        if not reader.can_read(8):
            break

        # Read property name (FName: index + number)
        name_idx = reader.read_int32()
        _name_num = reader.read_int32()
        name = name_map[name_idx] if 0 <= name_idx < len(name_map) else f"#{name_idx}"

        if name == "None":
            break

        count += 1

        if use_new_format:
            # New format: FPropertyTypeName + Size + Flags byte
            type_name = _read_property_type_name(reader, name_map)
            size = reader.read_int32()
            flags = reader.read_uint8()

            # HasArrayIndex
            if flags & TAG_HasArrayIndex:
                reader.skip(4)

            # Skip value data
            if type_name == "BoolProperty":
                # BoolProperty has Size=0, value is in flags
                pass
            else:
                reader.skip(size)

            # HasPropertyGuid
            if flags & TAG_HasPropertyGuid:
                reader.skip(16)

            # HasPropertyExtensions
            if use_extensions and (flags & TAG_HasPropertyExtensions):
                ext = reader.read_uint8()
                if ext & 0x01:  # OverridableInformation
                    reader.skip(1 + 4)  # OverrideOperation + bExperimentalOverridableLogic
        else:
            # Old format: FName type + Size + ArrayIndex + type-specific header
            type_idx = reader.read_int32()
            _type_num = reader.read_int32()
            type_name = name_map[type_idx] if 0 <= type_idx < len(name_map) else f"#{type_idx}"

            size = reader.read_int32()
            _array_index = reader.read_int32()

            # Type-specific header data
            if type_name == "StructProperty":
                reader.skip(8)   # StructName FName
                reader.skip(16)  # StructGuid
            elif type_name == "BoolProperty":
                reader.skip(1)   # BoolVal byte
                has_guid = reader.read_uint8()
                if has_guid:
                    reader.skip(16)
                continue  # BoolProperty has Size=0
            elif type_name in ("ByteProperty", "EnumProperty"):
                reader.skip(8)   # EnumName FName
            elif type_name == "ArrayProperty":
                reader.skip(8)   # InnerType FName
            elif type_name == "SetProperty":
                reader.skip(8)   # InnerType FName
            elif type_name == "MapProperty":
                reader.skip(16)  # InnerType + ValueType FNames

            has_guid = reader.read_uint8()
            if has_guid:
                reader.skip(16)

            # Skip value data
            reader.skip(size)

    return count
