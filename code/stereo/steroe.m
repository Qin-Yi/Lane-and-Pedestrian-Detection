videoFileLeft = 'left.avi';
videoFileRight = 'right.avi';

readerLeft = vision.VideoFileReader(videoFileLeft, 'VideoOutputDataType', 'uint8');
readerRight = vision.VideoFileReader(videoFileRight, 'VideoOutputDataType', 'uint8');
player = vision.DeployableVideoPlayer('Location', [20, 400]);

peopleDetector = vision.PeopleDetector('MinSize', [166 83]);

frame_index = 0;
dist_list = [];
while ~isDone(readerLeft) && ~isDone(readerRight)
    % Read the frames.
    frameLeft = readerLeft.step();
    frameRight = readerRight.step();

    % Rectify the frames.
    [frameLeftRect, frameRightRect] = ...
        rectifyStereoImages(frameLeft, frameRight, stereoParams);

    % Convert to grayscale.
    frameLeftGray  = rgb2gray(frameLeftRect);
    frameRightGray = rgb2gray(frameRightRect);
    filename = strcat(strcat('fral',num2str(frame_index,'%03d')),'.png');
    imwrite(frameLeftRect,filename);

    % Compute disparity.
    disparityMap = disparity(frameLeftGray, frameRightGray);
    filename = strcat(strcat('disp',num2str(frame_index,'%03d')),'.png');
    imwrite(disparityMap,filename);

    % Reconstruct 3-D scene.
    points3D = reconstructScene(disparityMap, stereoParams);
    points3D = points3D ./ 1000;
    ptCloud = pointCloud(points3D, 'Color', frameLeftRect);

    % Detect people.
    index = find(result.index == frame_index);
    bboxes = [result.x(index), result.y(index), result.w(index), result.h(index)];
    centroids_x = (bboxes(1)+round(bboxes(3)*2/5):1:bboxes(1)+bboxes(3)*3/5)';
    centroids_y = (bboxes(2)+round(bboxes(4)*2/5):1:bboxes(2)+bboxes(4)*3/5)';
    centroids = zeros(size(centroids_x,1) * size(centroids_y,1),2);
    for i = 1 : size(centroids_x,1)
        for j = 1 : size(centroids_y,1)
            centroids(size(centroids_y,1)*(i-1)+j,1)=centroids_x(i);
            centroids(size(centroids_y,1)*(i-1)+j,2)=centroids_y(j);
        end
    end
    centroidsIdx = sub2ind(size(disparityMap), centroids(:, 2), centroids(:, 1));
    X = points3D(:, :, 1);
    Y = points3D(:, :, 2);
    Z = points3D(:, :, 3);
    centroids3D = [X(centroidsIdx), Y(centroidsIdx), Z(centroidsIdx)];
    centroids3D(any(isnan(centroids3D)'),:)=[];
    centroids3D(any(isinf(centroids3D)'),:)=[];
    centroids3D(centroids3D(:,3) > 40,:)=[];
    
    dists = mode(round(sqrt(sum(centroids3D .^ 2, 2))));
    dist_list(frame_index+1) = dists;
    labels = cell(1, numel(dists));
    for i = 1:numel(dists)
        labels{i} = sprintf('%0.2f meters', dists(i));
    end
    dispFrame = insertObjectAnnotation(frameLeftRect, 'rectangle', bboxes,labels);
    filename = strcat(strcat('final',num2str(frame_index,'%03d')),'.png');
    imwrite(dispFrame,filename);
    frame_index = frame_index + 1
end

% Clean up.
reset(readerLeft);
reset(readerRight);
release(player);