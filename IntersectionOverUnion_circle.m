%% Intersection over union (overlap ratio) of circles
% This code computes overlap (intersection over union) of two circles, counts number of true positives, false positives and false negatives as well as 
% it computes recall, precision, F1-score and area-under-curve (AUC). It also plots success rate at different overlap threshold ranging from 0 to 1. 
% From this overlap success plot, we computed the AUC. The detection quality is measured using mainly AUC and also using F1-score (F-score or F-measure).

% This example is a function with the main body at the top and helper routines in the form of nested functions below.
function IntersectionOverUnion_circle()  % Parent function

% Example
circle1 = [10.0, 10.0, 6.0];
circle2 = [8.0, 8.0, 8.0];

iou = circle_intersection_over_union(circle1, circle2);
disp(['Example: ', num2str(iou)])


% ========== Compute Recall, Precision, F1_score  and AUC ==============================================================

% This example is just for two frames ------ Can be extended for as many frames as we want!
% Here zero-padding may be necessary to make equal matrix dimensions
groundtruth1 = [8, 8, 6; 16, 16, 8; 30, 30, 10]; % For frame 1
detections1 = [10, 10, 6; 20, 20, 8; 5, 5, 4; 40, 30, 12];

groundtruth2 = [8, 8, 6; 16, 16, 8; 30, 30, 10];  % For frame 2
detections2 = [10, 10, 6; 20, 20, 8; 16, 16, 10; 40, 30, 12];

N_frames = 2;
dim1_det = max(size(detections1,1), size(detections2,1));
dim2_det = max(size(detections1,2), size(detections2,2));

dim1_gt = max(size(groundtruth1,1), size(groundtruth2,1));
dim2_gt = max(size(groundtruth1,2), size(groundtruth2,2));

groundtruth_total = zeros(dim1_gt,dim2_gt, N_frames); % combine all groundtruths of all frames, in this case 2 frames
groundtruth_total(:,:,1) = groundtruth1;
groundtruth_total(:,:,2) = groundtruth2;

detections_total = zeros(dim1_det,dim2_det, N_frames); % combine all detections of all frames, in this case 2 frames
detections_total(:,:,1) = detections1;
detections_total(:,:,2) = detections2;

detections_iou_total = [];
groundTruth_iou_total = [];
for d = 1:size(groundtruth_total, 3) % for each frame d
    detections_d = detections_total(:,:,d);
    groundtruth_d = groundtruth_total(:,:,d);
    detection_groundtruth_iou = zeros(length(detections_d), length(groundtruth_d));
    for gt = 1: length(groundtruth_d)
       for dt = 1:length(detections_d)
          det_circle = detections_d(dt,:);
          gt_circle = groundtruth_d(gt,:);
          iou = circle_intersection_over_union(det_circle, gt_circle);
          detection_groundtruth_iou(dt,gt) = iou;
       end 
    end
    sprintf('detection_groundtruth_iou: %s', num2str(detection_groundtruth_iou));
    detections_iou = max(detection_groundtruth_iou,1);
    groundTruth_iou = max(detection_groundtruth_iou,0);
    sprintf('detections_iou: %s', num2str(detections_iou));
    sprintf('groundtruth_iou: %s', num2str(groundTruth_iou));
    detections_iou_total = [detections_iou_total detections_iou];
    groundTruth_iou_total = [groundTruth_iou_total groundTruth_iou];
end

sprintf('detections_iou_total: %s', num2str(detections_iou_total));
sprintf('groundTruth_iou_total: %s', num2str(groundTruth_iou_total));



















 % This function computes overlap (intersection over union) of two circles
    function overlapRatio = circle_intersection_over_union(circle1, circle2)   % Nested function
   
        % The format of the circles is (either detection or ground truth) is [xc, yc, r]
        x1 = circle1(1);  y1 = circle1(2); r1 = circle1(3);
        x2 = circle2(1); y2 = circle2(2); r2 = circle2(3);
        d2 = (x2 - x1).^2 + (y2 - y1).^2;
        d = sqrt(d2);
        t = ((r1 + r2)^2 - d2) * (d2 - (r2 - r1)^2);
        if t > 0  % The circles overlap
           intersectArea = r1^2 * acos((r1^2 - r2^2 + d2) / (2 * d * r1)) +r2^2 * acos((r2^2 - r1^2 + d2) / (2 * d * r2)) -1 / 2 * sqrt(t);
        elseif d > r1 + r2 % The circles are disjoint
           intersectArea = 0;
        else % One circle is contained entirely within the other
           intersectArea = pi * min(r1, r2)^2;
        end
        circle1Area = pi*r1^2;
        circle2Area = pi*r2^2;

        % Divide the intersection by union of bboxA and bboxB: bboxA U bboxB = bboxA +  bboxB - intersectAB
        overlapRatio = intersectArea / (circle1Area + circle2Area - intersectArea);

        % Return the intersection over union value
    end

end
   
   
   
