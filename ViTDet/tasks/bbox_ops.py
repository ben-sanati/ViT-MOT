import torch

class BBoxOperations:
    def __init__(self, image_size):
        self.image_size = image_size

    def xywh_to_xyxy(self, xywh):
        # input is (N, # objects, 4)
        # x1 = x | y1 = y | x2 = w + x | y2 = h + y
        xyxy = xywh.clone()
        xyxy[..., 2] = xyxy[..., 2] + xyxy[..., 0]
        xyxy[..., 3] = xyxy[..., 3] + xyxy[..., 1]
        return xyxy

    def xyxy_to_xywh(self, xyxy):
        # input is (N, # objects, 4)
        # x1 = x | y1 = y | w = x2 - x | h = y2 - y
        xywh = xyxy.clone()
        xywh[..., 2] = xywh[..., 2] - xywh[..., 0]
        xywh[..., 3] = xywh[..., 3] - xywh[..., 1]
        return xywh

    def GetIoU(self, BBox_pred, BBox_true, debug=False):
        """
            Input: predicted bbox and ground-truth bbox, each of size (4)
            Output: IoU integer score
        """
        # order bbox (x1, y1, x2, y2)
        BBox_pred_x1 = torch.min(BBox_pred[0], BBox_pred[2])
        BBox_pred_y1 = torch.min(BBox_pred[1], BBox_pred[3])
        BBox_pred_x2 = torch.max(BBox_pred[0], BBox_pred[2])
        BBox_pred_y2 = torch.max(BBox_pred[1], BBox_pred[3])
        BBox_true_x1 = torch.min(BBox_true[0], BBox_true[2])
        BBox_true_y1 = torch.min(BBox_true[1], BBox_true[3])
        BBox_true_x2 = torch.max(BBox_true[0], BBox_true[2])
        BBox_true_y2 = torch.max(BBox_true[1], BBox_true[3])
        
        # intersection area
        # ============================================================
        # coordinates of the intersection area
        x1 = torch.max(BBox_pred_x1, BBox_true_x1) 
        y1 = torch.max(BBox_pred_y1, BBox_true_y1)
        x2 = torch.min(BBox_pred_x2, BBox_true_x2)
        y2 = torch.min(BBox_pred_y2, BBox_true_y2)

        intersection_area = (x2 - x1).clamp(0) * (y2 - y1).clamp(0)
        # ============================================================
        
        # prediction area
        BBox_pred_area = (abs(BBox_pred_x1 - BBox_pred_x2) * abs(BBox_pred_y1 - BBox_pred_y2))
        
        # true area 
        BBox_true_area = (abs(BBox_true_x1 - BBox_true_x2) * abs(BBox_true_y1 - BBox_true_y2))
        
        # union area
        union_area = BBox_pred_area + BBox_true_area - intersection_area + 1e-6

        if debug:
            print(f"Intersection Area: {intersection_area}\tUnion Area: {union_area}")
        
        return intersection_area / union_area
    
    def NormalizeBBox(self, bbox_coord):
        # normalizes bbox coordinates to be between 0 and 1
        return bbox_coord / self.image_size

    def UnNormalizeBBox(self, bbox_coord):
        # unnormalizes bbox coordinates from [0, 1] to [0, Image Size]
        return bbox_coord * self.image_size
