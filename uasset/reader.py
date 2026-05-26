"""Binary reader for UE .uasset files."""
import struct
from io import BytesIO
from typing import Union, List, Callable, Optional


class BinaryReader:
    """Wraps a byte stream with typed read methods for UE binary formats."""

    def __init__(self, source: Union[str, bytes, BytesIO]):
        if isinstance(source, str):
            with open(source, 'rb') as f:
                self._data = f.read()
            self._stream = BytesIO(self._data)
        elif isinstance(source, bytes):
            self._data = source
            self._stream = BytesIO(source)
        elif isinstance(source, BytesIO):
            self._stream = source
            self._data = source.getvalue()
        else:
            raise TypeError(f"Unsupported source type: {type(source)}")

    @property
    def data(self) -> bytes:
        return self._data

    def position(self) -> int:
        return self._stream.tell()

    def seek(self, pos: int) -> None:
        self._stream.seek(pos)

    def skip(self, n: int) -> None:
        self._stream.seek(self._stream.tell() + n)

    def can_read(self, n: int) -> bool:
        return self._stream.tell() + n <= len(self._data)

    def read_bytes(self, n: int) -> bytes:
        data = self._stream.read(n)
        if len(data) < n:
            raise EOFError(f"Expected {n} bytes, got {len(data)} at position {self._stream.tell() - len(data)}")
        return data

    def read_int8(self) -> int:
        return struct.unpack('<b', self.read_bytes(1))[0]

    def read_uint8(self) -> int:
        return struct.unpack('<B', self.read_bytes(1))[0]

    def read_int16(self) -> int:
        return struct.unpack('<h', self.read_bytes(2))[0]

    def read_uint16(self) -> int:
        return struct.unpack('<H', self.read_bytes(2))[0]

    def read_int32(self) -> int:
        return struct.unpack('<i', self.read_bytes(4))[0]

    def read_uint32(self) -> int:
        return struct.unpack('<I', self.read_bytes(4))[0]

    def read_int64(self) -> int:
        return struct.unpack('<q', self.read_bytes(8))[0]

    def read_uint64(self) -> int:
        return struct.unpack('<Q', self.read_bytes(8))[0]

    def read_float(self) -> float:
        return struct.unpack('<f', self.read_bytes(4))[0]

    def read_double(self) -> float:
        return struct.unpack('<d', self.read_bytes(8))[0]

    def read_bool(self) -> bool:
        return self.read_int32() != 0

    def read_fstring(self) -> str:
        length = self.read_int32()
        if length > 0:
            data = self.read_bytes(length)
            # Strip null terminator
            try:
                return data.rstrip(b'\x00').decode('utf-8')
            except UnicodeDecodeError:
                return data.rstrip(b'\x00').decode('latin-1')
        elif length < 0:
            char_count = -length
            data = self.read_bytes(char_count * 2)
            try:
                return data.rstrip(b'\x00\x00').decode('utf-16-le')
            except UnicodeDecodeError:
                return data.rstrip(b'\x00\x00').decode('latin-1')
        else:
            return ""

    def read_fname(self, name_map: List[str]) -> str:
        index = self.read_int32()
        number = self.read_int32()
        if 0 <= index < len(name_map):
            return name_map[index]
        return f"#{index}"

    def read_tarray(self, func: Callable) -> list:
        count = self.read_int32()
        return [func() for _ in range(count)]

    def read_guid(self) -> bytes:
        return self.read_bytes(16)
