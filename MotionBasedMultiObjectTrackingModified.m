readfrom = 'example4.mp4';
writeto = 'myvideo4_1.mp4';
[unqlines, tracks, coefstr, pts] = MotionBasedMultiObject(readfrom, writeto);
function [frame_lines, tracks, coefstr, pts] = ...
    MotionBasedMultiObject(read, write)
% Create System objects used for reading video, detecting moving objects,
% and displaying the results.
obj = setupSystemObjects();
tracks = initializeTracks(); % Create an empty array of tracks.
nextId = 1; % ID of the next track
video = VideoWriter(write, 'MPEG-4');
video.FrameRate = obj.reader.FrameRate;
numFrames = obj.reader.NumFrames;
frame_lines = repmat(struct,1,numFrames);
coefstr = cell(numFrames, 1);
% Detect moving objects, and track them across video frames.
open(video);
frame_count = 1;
while hasFrame(obj.reader)
% while frame_count < 60
    
    frame = readFrame(obj.reader);
    [out, bboxes, centroids, UnqLines, coef, pts] = improc();
    coefstr{frame_count} = coef;
    frame_lines(frame_count).unqlines = UnqLines;
    Isub = imsubtract(out(:,:,2), rgb2gray(out));
    mask = imbinarize(Isub); 
    predictNewLocationsOfTracks();
    [assignments, unassignedTracks, unassignedDetections] = ...
        detectionToTrackAssignment();
    updateAssignedTracks();
    updateUnassignedTracks();
    deleteLostTracks();
    createNewTracks();
    displayTrackingResults();
    writeVideo(video, frame);
    frame_count = frame_count + 1;
end
close(video);

%% Image processing
function [out, bboxes, centroids, UnqLines, coef, pts] = improc()
    Img = rgb2gray(frame);
    % set of filters
%     Img = imadjust(Img, [0.3 0.6]);
%     Img = imsharpen(Img, 'Amount',1.2);
%     Img = medfilt2(Img, [3 3]);
%     Img = imdiffusefilt(Img, 'NumberOfIterations', 10);
%     Img = imguidedfilter(Img);
%     Img = fibermetric(Img, 'ObjectPolarity', 'dark');
%     BW = imbinarize(Img, 0.2);
%     BW = imbinarize(Img);
    Img = imgaussfilt(Img, 3.5);
    out = edge(Img, 'Canny', [0.02 0.2]);
%     out = imclose(out, 25);
%     out2 = edge(rgb2gray(frame), 'Sobel');
    
    % Finding straigth lines using Hough transform
    [H,T,R] = hough(out, 'RhoResolution', 1);
%     P  = houghpeaks(H, 3, 'threshold', ceil(0.3*max(H(:))));
%     lines = houghlines(BW, T, R, P, 'FillGap', 5, 'MinLength', 7);
    P  = houghpeaks(H, 5, 'threshold', ceil(0.35*max(H(:))));
    UnqLines = houghlines(out, T, R, P);
    out = uint8(repmat(out, 1, 1, 3)) .* 255;
    
%     % Filtering low area bounding boxes
%     numLin = length(UnqLines);
%     missInd = [];
%     indArray = 1:numLin;
%     bbtmp = ones(numLin, 4);
%     
%     for k = indArray
%         xy = [UnqLines(k).point1; UnqLines(k).point2];      
%         bbtmp(k, :) = round([xy(1, 1) max([xy(1, 2) xy(2, 2)])...
%             (xy(2, 1) - xy(1, 1)) abs(xy(2, 2) - xy(1, 2))]);
%         tmpSum = bbtmp(k, 3) + bbtmp(k, 4); % Vertical lines filtering
%         if tmpSum < 50 % Elminating small bboxes
%             missInd(end + 1) = k;
%         end
%     end
%     newInd = setdiff(indArray, missInd);
%     UnqLines = UnqLines(1,newInd);

    
%     % Checking for points on the same line -----> needs to be corrected
%     findEqMat = [newlines.theta; newlines.rho]';
%     [u, I, ~] = unique(findEqMat, 'rows', 'first');
%     hasDuplicates = size(u, 1) < size(findEqMat, 1);
%     if hasDuplicates
%         ixDupRows = setdiff(1:size(findEqMat,1), I);
%         dupRowValues = unique(findEqMat(ixDupRows, :), 'rows');
%         sameInd = cell(1, size(dupRowValues, 1));
%         for kk = 1:size(dupRowValues, 1)
%             [tmp, ~] = find(findEqMat == dupRowValues(kk,:));
%             sameInd{1, kk} = unique(tmp');
%         end
%             uniqInd = setdiff(1:length(newlines), ...
%                 unique(cell2mat(sameInd)));
%             UnqLines = newlines(1, uniqInd);         
%         for j=1:length(sameInd)
%             store1 = ones(1, length(sameInd{j})); 
%             store2 = ones(1, length(sameInd{j}));
%             for k = 1:length(sameInd{j})
%                 store1(k) = newlines(sameInd{j}(k)).point1(1);
%                 store2(k) = newlines(sameInd{j}(k)).point2(1);
%             end
%             [~, index1] = min(store1);
%             [~, index2] = max(store2);
%             newlines(sameInd{j}(index2)).point1 = ...
%                 newlines(sameInd{j}(index1)).point1;
%             UnqLines(end + 1) = newlines(sameInd{j}(index2));         
%         end
%     else
%         UnqLines = newlines; % no repetion case
%     end
    
    % Calculating bounding boxes, centroids and stright line coefficients
    numUnqLines = length(UnqLines);
    bboxes = ones(numUnqLines, 4);
    centroids = ones(numUnqLines, 2);
    coef = ones(numUnqLines, 2);
    point1 = reshape([UnqLines.point1], 2, [])';
    point2 = reshape([UnqLines.point2], 2, [])';
    pts = ones(numUnqLines, 4);
    mnan = [];
    for m = 1:numUnqLines
        if point1(m, 1) == point2(m, 1)
            coef(m, :) = [NaN, point1(m, 1)];
            mnan(end+1) = m;
        else
            coef(m, :) = polyfit([point1(m, 1), point2(m, 1)], ...
                [point1(m, 2), point2(m, 2)], 1);
        end
        pts(m, :) = lineToBorderPoints([coef(m, 1), -1, coef(m, 2)], ...
            size(Img));
        bboxes(m, :) = round([pts(m, 1), ...
            max([pts(m, 2) pts(m, 4)])...
            (pts(m, 3) - pts(m, 1)), ...
            abs(pts(m, 4) - pts(m, 2))]);
        centroids(m, :) = round([pts(m, 1) + (pts(m, 3) - pts(m, 1))/2, ...
            min([pts(m, 2) pts(m, 4)]) + abs(pts(m, 4) - pts(m, 2))/2]);
    end

    if ~isempty(mnan)
        for mn = mnan
            bboxes(mnan(mn), :) = [round(coef(mn, 2)), size(Img, 1), ...
                0, size(Img, 1)];
            centroids(m, :) = round([coef(mn, 2), ...
                size(Img, 1)/2]);
%         out = insertShape(out, 'Line', [coef(mn, 2), 0, ...
%             coef(mn, 2) size(Img, 1)], ...
%             'LineWidth', 3, 'Color', 'green', 'SmoothEdges', false);
        end
        pts = pts(~mnan, :);
    end
    out = insertShape(out, 'Line', [point1 point2], ...
                'LineWidth', 3, 'Color', 'green', 'SmoothEdges', false);

    
end
%% Initialize Video I/O 
 function obj = setupSystemObjects()
     % Create a video reader.
     obj.reader = VideoReader(read);
     
     % Create two video players, one to display the video,
     % and one to display the foreground mask.
     obj.maskPlayer = vision.VideoPlayer('Position', [740, 400, 700, 400]);
     obj.videoPlayer = vision.VideoPlayer('Position', [20, 400, 700, 400]);
 end
%% Creating an empty array of tracks
function tracks = initializeTracks()
    tracks = struct(...
        'id', {}, ...
        'bbox', {}, ...
        'kalmanFilter', {}, ...
        'age', {}, ...
        'totalVisibleCount', {}, ...
        'consecutiveInvisibleCount', {});
end
%% Predicting new locations of tracks
function predictNewLocationsOfTracks()
    for i = 1:length(tracks)
        bbox = tracks(i).bbox;
        % Predict the current location of the track.
        predictedCentroid = predict(tracks(i).kalmanFilter);
        
        % Shift the bounding box so that its center is at
        % the predicted location.
        tracks(i).bbox = [predictedCentroid(1) - bbox(3) / 2, ...
           predictedCentroid(2) - bbox(4) / 2, bbox(3:4)];
    end
end
%% Track detection
 function [assignments, unassignedTracks, unassignedDetections] = ...
            detectionToTrackAssignment()
        nTracks = length(tracks);
        nDetections = size(centroids, 1);
        
        % Compute the cost of assigning each detection to each track.
        cost = zeros(nTracks, nDetections);
        for i = 1:nTracks
            cost(i, :) = distance(tracks(i).kalmanFilter, centroids);
        end
        
        % Solve the assignment problem.
        costOfNonAssignment = 50;
        [assignments, unassignedTracks, unassignedDetections] = ...
            assignDetectionsToTracks(cost, costOfNonAssignment);
 end
%% Updating tracks
function updateAssignedTracks()
    numAssignedTracks = size(assignments, 1);
    for i = 1:numAssignedTracks
        trackIdx = assignments(i, 1);
        detectionIdx = assignments(i, 2);
        centroid = centroids(detectionIdx, :);
        bbox = bboxes(detectionIdx, :);
        
        % Correct the estimate of the object's location
        % using the new detection.
        correct(tracks(trackIdx).kalmanFilter, centroid);
        
        % Replace predicted bounding box with detected
        % bounding box.
        tracks(trackIdx).bbox = bbox;
        
        % Update track's age.
        tracks(trackIdx).age = tracks(trackIdx).age + 1;
        
        % Update visibility.
        tracks(trackIdx).totalVisibleCount = ...
            tracks(trackIdx).totalVisibleCount + 1;
        tracks(trackIdx).consecutiveInvisibleCount = 0;
    end
end
%% Life calculation of unasigned tracks
function updateUnassignedTracks()
    for i = 1:length(unassignedTracks)
        ind = unassignedTracks(i);
        tracks(ind).age = tracks(ind).age + 1;
        tracks(ind).consecutiveInvisibleCount = ...
            tracks(ind).consecutiveInvisibleCount + 1;
    end
end
%% Erasing lost tracks
function deleteLostTracks()
        if isempty(tracks)
            return;
        end
        invisibleForTooLong = 30;
        ageThreshold = 20;

        % Compute the fraction of the track's age for which it was visible.
        ages = [tracks(:).age];
        totalVisibleCounts = [tracks(:).totalVisibleCount];
        visibility = totalVisibleCounts ./ ages;

        % Find the indices of 'lost' tracks.
        lostInds = (ages < ageThreshold & visibility < 0.6) | ...
            [tracks(:).consecutiveInvisibleCount] >= invisibleForTooLong;

        % Delete lost tracks.
        tracks = tracks(~lostInds);
end
%%
function createNewTracks()
        centroids = centroids(unassignedDetections, :);
        bboxes = bboxes(unassignedDetections, :);
        for i = 1:size(centroids, 1)
            centroid = centroids(i,:);
            bbox = bboxes(i, :);

            % Create a Kalman filter object.
            kalmanFilter = configureKalmanFilter('ConstantAcceleration',...
                centroid, [200 1 0.1], [200, 1, 0.1], 1);

            % Create a new track.
            newTrack = struct(...
                'id', nextId, ...
                'bbox', bbox, ...
                'kalmanFilter', kalmanFilter, ...
                'age', 1, ...
                'totalVisibleCount', 1, ...
                'consecutiveInvisibleCount', 0);

            % Add it to the array of tracks.
            tracks(end + 1) = newTrack;

            % Increment the next id.
            nextId = nextId + 1;
        end
end
%%
function displayTrackingResults()
    % Convert the frame and the mask to uint8 RGB.
    frame = im2uint8(frame);
    mask = uint8(repmat(mask, [1, 1, 3])) .* 255;
    minVisibleCount = 10;
    if ~isempty(tracks)
        
        % Noisy detections tend to result in short-lived tracks.
        % Only display tracks that have been visible for more than
        % a minimum number of frames.
        reliableTrackInds = ...
            [tracks(:).totalVisibleCount] > minVisibleCount;
        reliableTracks = tracks(reliableTrackInds);
        
        % Display the objects. If an object has not been detected
        % in this frame, display its predicted bounding box.
        if ~isempty(reliableTracks)
            % Get bounding boxes.
            bboxes = cat(1, reliableTracks.bbox);
            
            % Get ids.
            ids = int32([reliableTracks(:).id]);
            
            % Create labels for objects indicating the ones for
            % which we display the predicted rather than the actual
            % location.
            labels = cellstr(int2str(ids'));
            predictedTrackInds = ...
                [reliableTracks(:).consecutiveInvisibleCount] > 10;
            isPredicted = cell(size(labels));
            isPredicted(predictedTrackInds) = {' predicted'};
            labels = strcat(labels, isPredicted);
            
            % Draw the objects on the frame.
            frame = insertObjectAnnotation(frame, 'circle', ...
                [bboxes(:, 1)+bboxes(:, 3)/2, ...
                bboxes(:, 2)-bboxes(:, 4)/2, ...
                10*ones(size(bboxes, 1), 1)], labels);
%             point1 = reshape([UnqLines.point1],2,[])';
%             point2 = reshape([UnqLines.point2],2,[])';
            frame = insertShape(frame, 'Line', pts, ...
                'LineWidth', 5, 'Color', 'red', 'SmoothEdges', false);
            
            
        end
    end
    
    % Display the mask and the frame.
    obj.maskPlayer.step(out);
%     Img = uint8(repmat(Img, [1, 1, 3]));
    obj.videoPlayer.step(frame);
end
end