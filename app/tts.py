import onnxruntime
from vits2_inference import utils
from .infer_onnx import synthesize_speech

class TTSInfer:
    def __init__(self, model_path, config_path, lang="en", sid=None, use_accent=True):
        self.model_path = model_path
        self.config_path = config_path
        self.lang = lang
        self.sid = sid
        self.use_accent = use_accent

        sess_options = onnxruntime.SessionOptions()
        self.model = onnxruntime.InferenceSession(
            str(self.model_path),
            sess_options=sess_options,
            providers=["CPUExecutionProvider"]
        )
        self.hps = utils.get_hparams_from_file(self.config_path)

    def synthesize(self, text, output_path):
        generation_time = synthesize_speech(
            text, self.model, self.hps, output_path, self.lang, self.sid, self.use_accent
        )
        return output_path, generation_time
    