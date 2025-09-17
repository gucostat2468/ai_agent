"""Plugin integration smoke tests."""

from ai_agent.plugins.examples.file_manager import FileManagerPlugin

def test_file_manager_list_read(tmp_path):
    root = tmp_path / "fmroot"
    root.mkdir()
    f = root / "a.txt"
    f.write_text("hello")
    fm = FileManagerPlugin(root=str(root))
    files = fm.list_files('.')
    assert 'a.txt' in files
    assert fm.read_file('a.txt') == "hello"
