from bitcal_tts.integrations.hf_inference import transformers_available


def test_transformers_flag_is_bool():
    assert isinstance(transformers_available(), bool)
