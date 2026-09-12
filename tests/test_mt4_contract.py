from pathlib import Path

from regime.microstructure.ingest import RECORD_SIZE, encode_header


EA_PATH = Path(__file__).resolve().parents[1] / "mt4" / "QuoteCaptureEA.mq4"


def test_binary_schema_matches_mt4_contract():
    assert len(encode_header()) == 8
    assert RECORD_SIZE == 84


def test_quote_capture_ea_remains_passive_only():
    source = EA_PATH.read_text(encoding="utf-8")
    forbidden = (
        "OrderSend(",
        "OrderClose(",
        "OrderModify(",
        "OrderDelete(",
        "OrderCloseBy(",
        "WebRequest(",
        "#import",
        "ShellExecute",
    )
    found = [token for token in forbidden if token in source]
    assert not found, f"passive collector contains forbidden capabilities: {found}"


def test_mt4_writer_declares_same_record_size():
    source = EA_PATH.read_text(encoding="utf-8")
    assert "#define SCHEMA_VERSION 1" in source
    assert "#define RECORD_SIZE 84" in source
    assert 'WriteByte(\'R\') && WriteByte(\'F\') && WriteByte(\'X\') && WriteByte(\'4\')' in source
