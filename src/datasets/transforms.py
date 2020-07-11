class CustomTransform(object):
    def __call__(self, input):
        return input['img']

    def __repr__(self):
        return self.__class__.__name__


class BoundingBoxCrop(CustomTransform):
    def __init__(self):
        pass

    def __call__(self, input):
        x, y, width, height = input['bbox'].long().tolist()
        left, top, right, bottom = x, y, x + width, y + height
        bboxc_img = input['img'].crop((left, top, right, bottom))
        return bboxc_img