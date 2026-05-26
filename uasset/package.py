"""Package header parser for UE5 .uasset files.

Based on reverse-engineered format from UAssetAPI source code and actual UE5.5 files.
Key difference from UE4: FileVersionUE5 comes BEFORE FileVersionLicenseeUE4.
"""
import struct
from typing import List, Optional, Tuple
from .reader import BinaryReader


# UE4 ObjectVersion enum values (from UAssetAPI ObjectVersion.cs)
VER_UE4_OLDEST_LOADABLE_PACKAGE = 214
VER_UE4_WORLD_LEVEL_INFO = 223
VER_UE4_CHANGED_CHUNKID_TO_BE_AN_ARRAY_OF_CHUNKIDS = 324
VER_UE4_ENGINE_VERSION_OBJECT = 334
VER_UE4_LOAD_FOR_EDITOR_GAME = 363
VER_UE4_ADD_STRING_ASSET_REFERENCES_MAP = 382
VER_UE4_PACKAGE_SUMMARY_HAS_COMPATIBLE_ENGINE_VERSION = 442
VER_UE4_SERIALIZE_TEXT_IN_PACKAGES = 457
VER_UE4_COOKED_ASSETS_IN_EDITOR_SUPPORT = 483
VER_UE4_TemplateIndex_IN_COOKED_EXPORTS = 506
VER_UE4_PRELOAD_DEPENDENCIES_IN_COOKED_EXPORTS = 505
VER_UE4_ADDED_SEARCHABLE_NAMES = 508
VER_UE4_64BIT_EXPORTMAP_SERIALSIZES = 509
VER_UE4_ADDED_PACKAGE_SUMMARY_LOCALIZATION_ID = 514
VER_UE4_ADDED_PACKAGE_OWNER = 516
VER_UE4_NON_OUTER_PACKAGE_IMPORT = 518

# UE5 ObjectVersionUE5 enum values (from UAssetAPI ObjectVersion.cs)
UE5_INITIAL_VERSION = 1000
UE5_NAMES_REFERENCED_FROM_EXPORT_DATA = 1001
UE5_PAYLOAD_TOC = 1002
UE5_OPTIONAL_RESOURCES = 1003
UE5_LARGE_WORLD_COORDINATES = 1004
UE5_REMOVE_OBJECT_EXPORT_PACKAGE_GUID = 1005
UE5_TRACK_OBJECT_EXPORT_IS_INHERITED = 1006
UE5_FSOFTOBJECTPATH_REMOVE_ASSET_PATH_FNAMES = 1007
UE5_ADD_SOFTOBJECTPATH_LIST = 1008
UE5_DATA_RESOURCES = 1009
UE5_SCRIPT_SERIALIZATION_OFFSET = 1010
UE5_PROPERTY_TAG_EXTENSION = 1011
UE5_PROPERTY_TAG_COMPLETE_TYPE_NAME = 1012
UE5_METADATA_SERIALIZATION_OFFSET = 1014
UE5_VERSE_CELLS = 1015
UE5_PACKAGE_SAVED_HASH = 1016
UE5_IMPORT_TYPE_HIERARCHIES = 1018

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
    """Parses a UE5 .uasset package file."""

    def __init__(self, filepath: str):
        self.filepath = filepath
        self.reader = BinaryReader(filepath)

        # Header fields
        self.tag = 0
        self.legacy_file_version = 0
        self.legacy_ue3_version = 0
        self.file_version_ue4 = 0
        self.file_version_ue5 = 0
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

        # 2. LegacyFileVersion
        self.legacy_file_version = r.read_int32()
        if self.legacy_file_version != -8:
            raise ValueError(f"Unexpected LegacyFileVersion: {self.legacy_file_version}")

        # 3. LegacyUE3Version
        self.legacy_ue3_version = r.read_int32()

        # 4. FileVersionUE4
        self.file_version_ue4 = r.read_int32()

        # 5. FileVersionUE5 (BEFORE FileVersionLicenseeUE4!)
        if self.legacy_file_version <= -8:
            self.file_version_ue5 = r.read_int32()

        # 6. FileVersionLicenseeUE4
        self.file_version_licensee_ue4 = r.read_int32()

        # 7. SavedHash + SectionSixOffset (UE5 >= PACKAGE_SAVED_HASH)
        if self.file_version_ue5 >= UE5_PACKAGE_SAVED_HASH:
            r.skip(20)  # SavedHash (FIoHash, 20 bytes)
            self.total_header_size = r.read_int32()  # SectionSixOffset

        # 8. Custom version container (Optimized format)
        if self.legacy_file_version <= -2:
            cv_count = r.read_int32()
            for _ in range(cv_count):
                guid = r.read_bytes(16)
                version = r.read_int32()
                self.custom_versions.append((guid, version))

        # 9. TotalHeaderSize / SectionSixOffset (UE5 < PACKAGE_SAVED_HASH)
        if self.file_version_ue5 < UE5_PACKAGE_SAVED_HASH:
            self.total_header_size = r.read_int32()

        # 10. FolderName
        self.folder_name = r.read_fstring()

        # 11. PackageFlags
        self.package_flags = r.read_uint32()

        # 12-13. NameCount, NameOffset
        self.name_count = r.read_int32()
        self.name_offset = r.read_int32()

        # 14. SoftObjectPaths (UE5 >= ADD_SOFTOBJECTPATH_LIST)
        if self.file_version_ue5 >= UE5_ADD_SOFTOBJECTPATH_LIST:
            _sop_count = r.read_int32()
            _sop_offset = r.read_int32()

        # 15. LocalizationId (ObjectVersion >= VER_UE4_ADDED_PACKAGE_SUMMARY_LOCALIZATION_ID)
        if self.file_version_ue4 >= VER_UE4_ADDED_PACKAGE_SUMMARY_LOCALIZATION_ID:
            _loc_id = r.read_fstring()

        # 16. GatherableTextData (ObjectVersion >= VER_UE4_SERIALIZE_TEXT_IN_PACKAGES)
        if self.file_version_ue4 >= VER_UE4_SERIALIZE_TEXT_IN_PACKAGES:
            _gather_count = r.read_int32()
            _gather_offset = r.read_int32()

        # 17-18. ExportCount, ExportOffset
        self.export_count = r.read_int32()
        self.export_offset = r.read_int32()

        # 19-20. ImportCount, ImportOffset
        self.import_count = r.read_int32()
        self.import_offset = r.read_int32()

        # 21. CellExport/Import (UE5 >= VERSE_CELLS)
        if self.file_version_ue5 >= UE5_VERSE_CELLS:
            r.skip(16)  # 4 int32s: CellExportCount, CellExportOffset, CellImportCount, CellImportOffset

        # 22. MetaDataOffset (UE5 >= METADATA_SERIALIZATION_OFFSET)
        if self.file_version_ue5 >= UE5_METADATA_SERIALIZATION_OFFSET:
            _meta_offset = r.read_int32()

        # 23. DependsOffset
        self.depends_offset = r.read_int32()

        # 24-25. SoftPackageReferences
        if self.file_version_ue4 >= VER_UE4_ADD_STRING_ASSET_REFERENCES_MAP:
            _soft_count = r.read_int32()
            _soft_offset = r.read_int32()

        # 26. SearchableNamesOffset
        if self.file_version_ue4 >= VER_UE4_ADDED_SEARCHABLE_NAMES:
            _search_offset = r.read_int32()

        # 27. ThumbnailTableOffset
        _thumb_offset = r.read_int32()

        # 28. ImportTypeHierarchies (UE5 >= IMPORT_TYPE_HIERARCHIES)
        if self.file_version_ue5 >= UE5_IMPORT_TYPE_HIERARCHIES:
            _ith_count = r.read_int32()
            _ith_offset = r.read_int32()

        # 29. PackageGuid (UE5 < PACKAGE_SAVED_HASH)
        if self.file_version_ue5 < UE5_PACKAGE_SAVED_HASH:
            _pkg_guid = r.read_bytes(16)

        # 30. PersistentGuid (ObjectVersion >= VER_UE4_ADDED_PACKAGE_OWNER)
        if self.file_version_ue4 >= VER_UE4_ADDED_PACKAGE_OWNER:
            _persistent_guid = r.read_bytes(16)

        # 31. Extra bytes (VER_UE4_ADDED_PACKAGE_OWNER <= ObjectVersion < VER_UE4_NON_OUTER_PACKAGE_IMPORT)
        if (self.file_version_ue4 >= VER_UE4_ADDED_PACKAGE_OWNER and
                self.file_version_ue4 < VER_UE4_NON_OUTER_PACKAGE_IMPORT):
            r.skip(16)

        # 32-33. Generations
        gen_count = r.read_int32()
        for _ in range(gen_count):
            r.skip(8)  # ExportCount + NameCount per generation

        # 34. SavedByEngineVersion (ObjectVersion >= VER_UE4_ENGINE_VERSION_OBJECT)
        if self.file_version_ue4 >= VER_UE4_ENGINE_VERSION_OBJECT:
            self._read_engine_version(r)

        # 35. CompatibleWithEngineVersion
        if self.file_version_ue4 >= VER_UE4_PACKAGE_SUMMARY_HAS_COMPATIBLE_ENGINE_VERSION:
            self._read_engine_version(r)

        # 36. CompressionFlags
        _comp_flags = r.read_uint32()

        # 37. CompressedChunksCount
        chunk_count = r.read_int32()
        if chunk_count > 0:
            r.skip(chunk_count * 16)

        # 38. PackageSource
        _pkg_source = r.read_uint32()

        # 39. AdditionalPackagesToCook
        add_count = r.read_int32()
        for _ in range(add_count):
            r.read_fstring()

        # 40. TextureAllocations (LegacyFileVersion > -7) — not for our files (-8)

        # 41. AssetRegistryDataOffset
        _asset_reg = r.read_int32()

        # 42. BulkDataStartOffset
        self.bulk_data_start_offset = r.read_int64()

        # 43. WorldTileInfoDataOffset (ObjectVersion >= VER_UE4_WORLD_LEVEL_INFO)
        if self.file_version_ue4 >= VER_UE4_WORLD_LEVEL_INFO:
            _world_tile = r.read_int32()

        # 44. ChunkIDs (ObjectVersion >= VER_UE4_CHANGED_CHUNKID_TO_BE_AN_ARRAY_OF_CHUNKIDS)
        if self.file_version_ue4 >= VER_UE4_CHANGED_CHUNKID_TO_BE_AN_ARRAY_OF_CHUNKIDS:
            chunk_id_count = r.read_int32()
            r.skip(chunk_id_count * 4)

        # 45-46. PreloadDependency
        if self.file_version_ue4 >= VER_UE4_PRELOAD_DEPENDENCIES_IN_COOKED_EXPORTS:
            _preload_count = r.read_int32()
            _preload_offset = r.read_int32()

        # 47. NamesReferencedFromExportDataCount (UE5 >= NAMES_REFERENCED_FROM_EXPORT_DATA)
        if self.file_version_ue5 >= UE5_NAMES_REFERENCED_FROM_EXPORT_DATA:
            _names_ref_count = r.read_int32()

        # 48. PayloadTocOffset (UE5 >= PAYLOAD_TOC)
        if self.file_version_ue5 >= UE5_PAYLOAD_TOC:
            _payload_toc_offset = r.read_int64()

        # 49. DataResourceOffset (UE5 >= DATA_RESOURCES)
        if self.file_version_ue5 >= UE5_DATA_RESOURCES:
            _data_resource_offset = r.read_int32()

        # Now read maps
        self._read_name_map()
        self._read_import_map()
        self._read_export_map()

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
        has_optional = self.file_version_ue5 >= UE5_OPTIONAL_RESOURCES
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

            if has_optional:
                entry.b_import_optional = r.read_int32() != 0

            self.imports.append(entry)

    def _read_export_map(self):
        r = self.reader
        r.seek(self.export_offset)

        has_template = self.file_version_ue4 >= VER_UE4_TemplateIndex_IN_COOKED_EXPORTS
        has_64bit_serial = self.file_version_ue4 >= VER_UE4_64BIT_EXPORTMAP_SERIALSIZES
        no_pkg_guid = self.file_version_ue5 >= UE5_REMOVE_OBJECT_EXPORT_PACKAGE_GUID
        has_inherited = self.file_version_ue5 >= UE5_TRACK_OBJECT_EXPORT_IS_INHERITED
        has_optional = self.file_version_ue5 >= UE5_OPTIONAL_RESOURCES
        has_script_offset = self.file_version_ue5 >= UE5_SCRIPT_SERIALIZATION_OFFSET
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

            if not no_pkg_guid:
                r.skip(16)  # PackageGuid

            if has_inherited:
                entry.is_inherited_instance = r.read_int32()

            entry.package_flags = r.read_uint32()

            if has_editor_game:
                entry.b_not_always_loaded_for_editor_game = r.read_int32()

            if has_is_asset:
                entry.b_is_asset = r.read_int32()

            if has_optional:
                entry.generate_public_hash = r.read_int32()

            if has_preload:
                entry.first_export_dependency = r.read_int32()
                entry.serialization_before_serialization_dependencies = r.read_int32()
                entry.create_before_serialization_dependencies = r.read_int32()
                entry.serialization_after_serialization_dependencies = r.read_int32()
                entry.create_before_create_dependencies = r.read_int32()

            if has_script_offset:
                entry.script_serialization_start_offset = r.read_int64()
                entry.script_serialization_end_offset = r.read_int64()

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
