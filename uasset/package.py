"""Package header parser for UE4 .uasset files.

Based on reverse-engineered format from UAssetAPI source code and actual UE4.27 files.
Key features:
- LegacyFileVersion -7 for UE4.26/4.27
- No FileVersionUE5 field (UE5 only)
- FileVersionUE4 517 for UE4.27
"""
import struct
from typing import List, Optional, Tuple
from .reader import BinaryReader


# UE4 ObjectVersion enum values — verified against the UE4.27.2 source in
# UnrealEngine-4.27.2-release/Engine/Source/Runtime/Core/Public/UObject/ObjectVersion.h
# (EUnrealEngineObjectUE4Version auto-increments from VER_UE4_OLDEST_LOADABLE_PACKAGE=214).
# IMPORTANT: a previous revision had these off-by-one (copied from an older UE4 enum
# where one entry before WORLD_LEVEL_INFO had not yet been inserted). Corrected to
# match UE4.27 so FileVersionUE4==514 assets parse: the LocalizationId field is
# serialized only when FileVersionUE4 >= 515 (see PackageFileSummary.cpp:196-203).
VER_UE4_OLDEST_LOADABLE_PACKAGE = 214
VER_UE4_WORLD_LEVEL_INFO = 224

# UE5 property tag feature flags (not used in UE4.27, values provided for compatibility)
UE5_PROPERTY_TAG_COMPLETE_TYPE_NAME = 1008
UE5_PROPERTY_TAG_EXTENSION = 1011
VER_UE4_CHANGED_CHUNKID_TO_BE_AN_ARRAY_OF_CHUNKIDS = 325
VER_UE4_ENGINE_VERSION_OBJECT = 335
VER_UE4_LOAD_FOR_EDITOR_GAME = 364
VER_UE4_ADD_STRING_ASSET_REFERENCES_MAP = 383
VER_UE4_PACKAGE_SUMMARY_HAS_COMPATIBLE_ENGINE_VERSION = 443
VER_UE4_SERIALIZE_TEXT_IN_PACKAGES = 458
VER_UE4_COOKED_ASSETS_IN_EDITOR_SUPPORT = 484
VER_UE4_TemplateIndex_IN_COOKED_EXPORTS = 507
VER_UE4_PRELOAD_DEPENDENCIES_IN_COOKED_EXPORTS = 506
VER_UE4_ADDED_SEARCHABLE_NAMES = 509
VER_UE4_64BIT_EXPORTMAP_SERIALSIZES = 510
VER_UE4_ADDED_PACKAGE_SUMMARY_LOCALIZATION_ID = 515
VER_UE4_ADDED_PACKAGE_OWNER = 517
VER_UE4_NON_OUTER_PACKAGE_IMPORT = 519
VER_UE4_517 = 517  # UE4.27 ObjectVersion

# Bulk data flags (from UE source, EBulkDataFlags)
BULKDATA_PayloadAtEndOfFile = 1 << 0       # 0x01
BULKDATA_SerializeCompressedZLIB = 1 << 1  # 0x02
BULKDATA_ForceSingleElementSerialization = 1 << 2  # 0x04
BULKDATA_SingleUse = 1 << 3                # 0x08
BULKDATA_Unused = 1 << 5                   # 0x20
BULKDATA_ForceInlinePayload = 1 << 6       # 0x40
BULKDATA_ForceStreamPayload = 1 << 7       # 0x80
BULKDATA_PayloadInSeperateFile = 1 << 8    # 0x100
BULKDATA_OptionalPayload = 1 << 11         # 0x800
BULKDATA_Size64Bit = 1 << 13              # 0x2000
BULKDATA_AtLargeOffsets = 1 << 17         # 0x200000


class ImportEntry:
    __slots__ = ('class_package', 'class_name', 'outer_index', 'object_name', 'b_import_optional')

    def __init__(self):
        self.class_package = ""
        self.class_name = ""
        self.outer_index = 0
        self.object_name = ""
        self.b_import_optional = False


class ExportEntry:
    __slots__ = (
        'class_index', 'super_index', 'template_index', 'outer_index',
        'object_name', 'object_flags', 'serial_size', 'serial_offset',
        'b_forced_export', 'b_not_for_client', 'b_not_for_server',
        'is_inherited_instance', 'package_flags',
        'b_not_always_loaded_for_editor_game', 'b_is_asset',
        'generate_public_hash', 'first_export_dependency',
        'serialization_before_serialization_dependencies',
        'create_before_serialization_dependencies',
        'serialization_after_serialization_dependencies',
        'create_before_create_dependencies',
        'script_serialization_start_offset',
        'script_serialization_end_offset',
    )

    def __init__(self):
        self.class_index = 0
        self.super_index = 0
        self.template_index = 0
        self.outer_index = 0
        self.object_name = ""
        self.object_flags = 0
        self.serial_size = 0
        self.serial_offset = 0
        self.b_forced_export = 0
        self.b_not_for_client = 0
        self.b_not_for_server = 0
        self.is_inherited_instance = 0
        self.package_flags = 0
        self.b_not_always_loaded_for_editor_game = 0
        self.b_is_asset = 0
        self.generate_public_hash = 0
        self.first_export_dependency = 0
        self.serialization_before_serialization_dependencies = 0
        self.create_before_serialization_dependencies = 0
        self.serialization_after_serialization_dependencies = 0
        self.create_before_create_dependencies = 0
        self.script_serialization_start_offset = 0
        self.script_serialization_end_offset = 0


class Package:
    """Parses a UE4 .uasset package file."""

    def __init__(self, filepath: str):
        self.filepath = filepath
        self.reader = BinaryReader(filepath)

        # Header fields
        self.tag = 0
        self.legacy_file_version = 0
        self.legacy_ue3_version = 0
        self.file_version_ue4 = 0
        self.file_version_ue5 = 0  # UE5 only, 0 for UE4.27
        self.file_version_licensee_ue4 = 0
        self.custom_versions: List[Tuple[bytes, int]] = []
        self.total_header_size = 0
        self.folder_name = ""
        self.package_flags = 0
        self.name_count = 0
        self.name_offset = 0
        self.export_count = 0
        self.export_offset = 0
        self.import_count = 0
        self.import_offset = 0
        self.depends_offset = 0
        self.bulk_data_start_offset = 0

        # Maps
        self.name_map: List[str] = []
        self.imports: List[ImportEntry] = []
        self.exports: List[ExportEntry] = []

        self._parse()

    def _parse(self):
        r = self.reader
        r.seek(0)

        # 1. Tag
        self.tag = r.read_uint32()
        if self.tag != 0x9E2A83C1:
            raise ValueError(f"Invalid tag: 0x{self.tag:08X}")

        # 2. LegacyFileVersion (UE4.27 uses -7)
        self.legacy_file_version = r.read_int32()
        if self.legacy_file_version != -7:
            raise ValueError(f"Unexpected LegacyFileVersion: {self.legacy_file_version} (expected -7 for UE4.27)")

        # 3. LegacyUE3Version
        self.legacy_ue3_version = r.read_int32()

        # 4. FileVersionUE4
        self.file_version_ue4 = r.read_int32()

        # 5. FileVersionLicenseeUE4
        self.file_version_licensee_ue4 = r.read_int32()

        # 6. Custom version container (Optimized format)
        if self.legacy_file_version <= -2:
            cv_count = r.read_int32()
            for _ in range(cv_count):
                guid = r.read_bytes(16)
                version = r.read_int32()
                self.custom_versions.append((guid, version))

        # 7. TotalHeaderSize
        self.total_header_size = r.read_int32()
        #print(f"  TotalHeaderSize: {self.total_header_size}, position: {r.position()}")

        # 8. FolderName
        self.folder_name = r.read_fstring()
        #print(f"  FolderName: '{self.folder_name}', position: {r.position()}")

        # 9. PackageFlags
        self.package_flags = r.read_uint32()
        #print(f"  PackageFlags: 0x{self.package_flags:08X}, position: {r.position()}")

        # 10-11. NameCount, NameOffset
        self.name_count = r.read_int32()
        self.name_offset = r.read_int32()
        #print(f"  NameCount: {self.name_count}, NameOffset: {self.name_offset}, position: {r.position()}")

        # 12. LocalizationId (ObjectVersion >= VER_UE4_ADDED_PACKAGE_SUMMARY_LOCALIZATION_ID)
        if self.file_version_ue4 >= VER_UE4_ADDED_PACKAGE_SUMMARY_LOCALIZATION_ID:
            _loc_id = r.read_fstring()

        # 13. GatherableTextData (ObjectVersion >= VER_UE4_SERIALIZE_TEXT_IN_PACKAGES)
        if self.file_version_ue4 >= VER_UE4_SERIALIZE_TEXT_IN_PACKAGES:
            _gather_count = r.read_int32()
            _gather_offset = r.read_int32()

        # 14-15. ExportCount, ExportOffset
        self.export_count = r.read_int32()
        self.export_offset = r.read_int32()
        #print(f"  ExportCount: {self.export_count}, ExportOffset: {self.export_offset}, position: {r.position()}")

        # 16-17. ImportCount, ImportOffset
        self.import_count = r.read_int32()
        self.import_offset = r.read_int32()
        #print(f"  ImportCount: {self.import_count}, ImportOffset: {self.import_offset}, position: {r.position()}")

        # 18. DependsOffset
        self.depends_offset = r.read_int32()
        #print(f"  DependsOffset: {self.depends_offset}, position: {r.position()}")

        # 19-20. SoftPackageReferences
        if self.file_version_ue4 >= VER_UE4_ADD_STRING_ASSET_REFERENCES_MAP:
            _soft_count = r.read_int32()
            _soft_offset = r.read_int32()

        # 21. SearchableNamesOffset
        if self.file_version_ue4 >= VER_UE4_ADDED_SEARCHABLE_NAMES:
            _search_offset = r.read_int32()

        # 22. ThumbnailTableOffset
        _thumb_offset = r.read_int32()

        # 23. PackageGuid
        _pkg_guid = r.read_bytes(16)

        # 24. PersistentGuid (ObjectVersion >= VER_UE4_ADDED_PACKAGE_OWNER) - SKIPPED for cooked packages
        # 25. OwnerPersistentGuid (VER_UE4_ADDED_PACKAGE_OWNER <= ObjectVersion < VER_UE4_NON_OUTER_PACKAGE_IMPORT) - SKIPPED for cooked packages
        # Note: These editor-only fields may not be present in some packages

        # 26-27. Generations
        gen_count = r.read_int32()
        #print(f"  Generations count: {gen_count}")
        # Safety check for corrupt gen_count
        if gen_count < 0 or gen_count > 1000:
            raise ValueError(f"gen_count out of range: {gen_count}")
        for i in range(gen_count):
            r.skip(8)  # ExportCount + NameCount per generation

        # 28. SavedByEngineVersion (ObjectVersion >= VER_UE4_ENGINE_VERSION_OBJECT)
        if self.file_version_ue4 >= VER_UE4_ENGINE_VERSION_OBJECT:
            self._read_engine_version(r)

        # 29. CompatibleWithEngineVersion
        if self.file_version_ue4 >= VER_UE4_PACKAGE_SUMMARY_HAS_COMPATIBLE_ENGINE_VERSION:
            self._read_engine_version(r)

        # 30. CompressionFlags
        _comp_flags = r.read_uint32()

        # 31. CompressedChunksCount
        chunk_count = r.read_int32()
        if chunk_count > 0:
            r.skip(chunk_count * 16)

        # 32. PackageSource
        _pkg_source = r.read_uint32()

        # 33. AdditionalPackagesToCook
        add_count = r.read_int32()
        for _ in range(add_count):
            r.read_fstring()

        # 34. TextureAllocations (only for LegacyFileVersion > -7)
        if self.legacy_file_version > -7:
            _texture_alloc_count = r.read_int32()
            if _texture_alloc_count > 0:
                r.skip(_texture_alloc_count * 12)

        # 35. AssetRegistryDataOffset
        _asset_reg = r.read_int32()

        # 36. BulkDataStartOffset
        self.bulk_data_start_offset = r.read_int64()

        # 37. WorldTileInfoDataOffset (ObjectVersion >= VER_UE4_WORLD_LEVEL_INFO)
        if self.file_version_ue4 >= VER_UE4_WORLD_LEVEL_INFO:
            _world_tile = r.read_int32()

        # 38. ChunkIDs (ObjectVersion >= VER_UE4_CHANGED_CHUNKID_TO_BE_AN_ARRAY_OF_CHUNKIDS)
        if self.file_version_ue4 >= VER_UE4_CHANGED_CHUNKID_TO_BE_AN_ARRAY_OF_CHUNKIDS:
            chunk_id_count = r.read_int32()
            r.skip(chunk_id_count * 4)

        # 39-40. PreloadDependency
        if self.file_version_ue4 >= VER_UE4_PRELOAD_DEPENDENCIES_IN_COOKED_EXPORTS:
            _preload_count = r.read_int32()
            _preload_offset = r.read_int32()

        #print("  Reading name map...")
        self._read_name_map()
        #print(f"  Name map read: {len(self.name_map)} names")
        
        #print("  Reading import map...")
        self._read_import_map()
        #print(f"  Import map read: {len(self.imports)} imports")
        
        #print("  Reading export map...")
        self._read_export_map()
        #print(f"  Export map read: {len(self.exports)} exports")

    @staticmethod
    def _read_engine_version(r: BinaryReader):
        r.skip(2 + 2 + 2 + 4)  # major, minor, patch, changelist
        r.read_fstring()  # branch

    def _read_name_map(self):
        r = self.reader
        r.seek(self.name_offset)
        self.name_map = []
        for _ in range(self.name_count):
            name = r.read_fstring()
            # Hash (4 bytes for version >= VER_UE4_NAME_HASHES_SERIALIZED)
            _hash = r.read_uint32()
            self.name_map.append(name)

    def _read_import_map(self):
        r = self.reader
        r.seek(self.import_offset)
        has_package_name = self.file_version_ue4 >= VER_UE4_NON_OUTER_PACKAGE_IMPORT

        self.imports = []
        for _ in range(self.import_count):
            entry = ImportEntry()
            cp_idx = r.read_int32()
            _cp_num = r.read_int32()
            cn_idx = r.read_int32()
            _cn_num = r.read_int32()
            entry.outer_index = r.read_int32()
            on_idx = r.read_int32()
            _on_num = r.read_int32()

            entry.class_package = self.name_map[cp_idx] if 0 <= cp_idx < len(self.name_map) else f"#{cp_idx}"
            entry.class_name = self.name_map[cn_idx] if 0 <= cn_idx < len(self.name_map) else f"#{cn_idx}"
            entry.object_name = self.name_map[on_idx] if 0 <= on_idx < len(self.name_map) else f"#{on_idx}"

            if has_package_name:
                r.skip(8)  # PackageName (FName: index + number)

            self.imports.append(entry)

    def _read_export_map(self):
        r = self.reader
        r.seek(self.export_offset)

        has_template = self.file_version_ue4 >= VER_UE4_TemplateIndex_IN_COOKED_EXPORTS
        has_64bit_serial = self.file_version_ue4 >= VER_UE4_64BIT_EXPORTMAP_SERIALSIZES
        has_preload = self.file_version_ue4 >= VER_UE4_PRELOAD_DEPENDENCIES_IN_COOKED_EXPORTS
        has_editor_game = self.file_version_ue4 >= VER_UE4_LOAD_FOR_EDITOR_GAME
        has_is_asset = self.file_version_ue4 >= VER_UE4_COOKED_ASSETS_IN_EDITOR_SUPPORT

        self.exports = []
        for _ in range(self.export_count):
            entry = ExportEntry()
            entry.class_index = r.read_int32()
            entry.super_index = r.read_int32()
            if has_template:
                entry.template_index = r.read_int32()
            entry.outer_index = r.read_int32()

            # ObjectName FName
            on_idx = r.read_int32()
            _on_num = r.read_int32()
            entry.object_name = self.name_map[on_idx] if 0 <= on_idx < len(self.name_map) else f"#{on_idx}"

            entry.object_flags = r.read_uint32()

            if has_64bit_serial:
                entry.serial_size = r.read_int64()
                entry.serial_offset = r.read_int64()
            else:
                entry.serial_size = r.read_int32()
                entry.serial_offset = r.read_int32()

            entry.b_forced_export = r.read_int32()
            entry.b_not_for_client = r.read_int32()
            entry.b_not_for_server = r.read_int32()

            # PackageGuid (always present in UE4)
            r.skip(16)

            entry.package_flags = r.read_uint32()

            if has_editor_game:
                entry.b_not_always_loaded_for_editor_game = r.read_int32()

            if has_is_asset:
                entry.b_is_asset = r.read_int32()

            if has_preload:
                entry.first_export_dependency = r.read_int32()
                entry.serialization_before_serialization_dependencies = r.read_int32()
                entry.create_before_serialization_dependencies = r.read_int32()
                entry.serialization_after_serialization_dependencies = r.read_int32()
                entry.create_before_create_dependencies = r.read_int32()

            self.exports.append(entry)

    def resolve_fname(self, index: int) -> str:
        if 0 <= index < len(self.name_map):
            return self.name_map[index]
        return f"#{index}"

    def get_export_class_name(self, export_index: int) -> str:
        if export_index < 0 or export_index >= len(self.exports):
            return "None"
        class_idx = self.exports[export_index].class_index
        if class_idx > 0:
            # Export reference
            exp_idx = class_idx - 1
            if 0 <= exp_idx < len(self.exports):
                return self.exports[exp_idx].object_name
            return f"Export[{exp_idx}]"
        elif class_idx < 0:
            # Import reference — use object_name (e.g. "StaticMesh") not class_name (e.g. "Class")
            imp_idx = -class_idx - 1
            if 0 <= imp_idx < len(self.imports):
                return self.imports[imp_idx].object_name
            return f"Import[{imp_idx}]"
        return "None"

    def get_export_data(self, export_index: int) -> Optional[BinaryReader]:
        if export_index < 0 or export_index >= len(self.exports):
            return None
        entry = self.exports[export_index]
        if entry.serial_size <= 0 or entry.serial_offset <= 0:
            return None
        self.reader.seek(entry.serial_offset)
        data = self.reader.read_bytes(entry.serial_size)
        return BinaryReader(data)

    def find_exports_by_class(self, class_name: str) -> List[int]:
        result = []
        for i, entry in enumerate(self.exports):
            cn = self.get_export_class_name(i)
            if cn == class_name:
                result.append(i)
        return result

    def resolve_export_name(self, export_index: int) -> str:
        if 0 <= export_index < len(self.exports):
            return self.exports[export_index].object_name
        return f"#{export_index}"
