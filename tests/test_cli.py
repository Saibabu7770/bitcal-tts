from bitcal_tts.cli import main


def test_cli_demo_runs(capsys):
    main(["demo", "--max-steps", "1", "--budget", "64"])
    out = capsys.readouterr().out
    assert "BitCal-TTS baseline demo" in out


def test_cli_doctor(capsys):
    main(["doctor"])
    out = capsys.readouterr().out
    assert "torch" in out.lower() or "not installed" in out.lower()
