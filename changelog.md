# Changelog

Changes in this fork of Prikalel/unreal-assets-to-glb.

## 2026-11-20

The exporter reads packages from UE 5.0 to 5.3 as well as 5.5: the older property tag layout, the SerializationControl byte only where it exists, inline source data in UE 5.0 packages that have no trailer, uncompressed buffers, and PNG texture sources that UE 5.2 marks only by their compression format. Texture colors come out right: UE stores 8-bit BGRA8 sources as PNGs without reordering the channels, so red and blue were swapped, and they are now swapped back (16-bit and greyscale sources are left alone). Base color parameters are found however a pack spells them. Textures are written to PNG one at a time and read back when a mesh embeds them, in place of the pickle cache that held every decoded texture in memory, so large packs no longer run out of RAM and an interrupted run resumes. `--skip-textures` works again: the PNGs go to a temporary folder that is removed afterwards, and the GLBs still embed their textures. The README lists the changes.
