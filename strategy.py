
class Strategy:
    def __init__(self,
            group_bbox_detector, unit_bbox_detector,
            text_recognizer, translator, inpainter, image_renderer
        ):
        self.group_bbox_detector = group_bbox_detector
        self.unit_bbox_detector = unit_bbox_detector
        self.translator = translator
        self.inpainter = inpainter
        self.image_renderer = image_renderer
        self.text_recognizer = text_recognizer

    def process(self, image):
        # Collect first set of boxes.

        group_boxes = self.group_bbox_detector(image)
        unit_boxes = self.unit_bbox_detector(image)
        texts = self.detect_texts(image, unit_boxes)
        collected = self._collect(group_boxes, unit_boxes, texts)
        image = self._inpaint(image, collected)

        # Collected is 
        #  [ group_bbox: {
        #        'units':
        #        'texts':
        #  }]

    def _translate(self, image):
        def _prettify(text):
            tgt_text = tgt_text.replace(' ', '')
            tgt_text = tgt_text.replace('▁', ' ')
        text = _prettify(v['text'])


    def _inpaint(self, image, collected):
        image_for_inpainting = image.copy()

        for group_bbox, v in collected.items():
            mask = np.zeros_like(image_for_inpainting)

            cv2.rectangle(
                    mask, 
                    (group_bbox.x, group_bbox.y), 
                    (group_bbox.X, group_bbox.Y), 
                    (255, 255, 255), 
                    thickness=cv2.FILLED
                )

            for bbox in v['units']:
                cv2.rectangle(
                        mask, 
                        (bbox.x, bbox.y), (bbox.X, bbox.Y), 
                        (255, 255, 255), thickness=cv2.FILLED
                )

        image_for_inpainting = self.inpainter(
                image_for_inpainting, 
                mask
        )

        return image_for_inpainting


    def detect_texts(self, image, unit_boxes):
        texts = []
        for i, bbox in enumerate(unit_boxes):
            cropped = image[bbox.y:bbox.Y, bbox.x:bbox.X, :]
            text = crnnw.predict(cropped)
            texts.append(text)
        return texts
