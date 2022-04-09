import os 
import cv2

from reg.vietocr.vietocr_infer import VietOCR
from det.db_paddle.infer import PaddleTextDetector
from utils.postprocessing import get_full_text_from_output, get_name_from_output

class Reader:
    def __init__(self, 
                det_name='DB', 
                det_config_path='config/ocr_det_db.yml',
                det_use_gpu=False, 
                reg_name='seq2seq', 
                reg_config_path='config//ocr_reg_seq2seq.yml',
                reg_use_gpu=False):

        if det_name == "DB":
            self.detect_model = PaddleTextDetector(config_path=det_config_path, use_gpu=det_use_gpu) 
        else:
            raise NotImplementedError("Model {} not implemented".format(det_name))

        if reg_name == 'seq2seq':
            self.recognize_model = VietOCR(config_path=reg_config_path, use_gpu=reg_use_gpu)
        else:
            raise NotImplementedError("Model {} not implemented".format(reg_name))
        
    def __call__(self, image, format_text=True):
        """
        Args:
            image (np.ndarray): BGR image

        Returns:
            list: ocr result - list of (polygon box, text, prob),
                    with polygon box format [[x1, y1], [x2, y2], [x3, y3], [x4, y4]]
        """
        text_boxes = self.detect_model(image)
        text_result = self.recognize_model.recognize(
            image=image,
            free_list=text_boxes,                                   
        )

        if format_text:
            output = {}
            full_text =  get_full_text_from_output(text_result)
            name = get_name_from_output(text_result)
            output['full_text'] = full_text 
            output['name'] = name
            return output 
        return text_result