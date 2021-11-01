readfrom = 'alpha2.mp4'; writeto = 'test2';
[unqlines, tracks, coefstr, pts, centrs] = ...
    MotionBasedMultiObject(readfrom, writeto);
function [frame_lines, tracks, coefstr, pts, centrs] = ...
    MotionBasedMultiObject(read, write)
% Create System objects used for reading video, detecting moving objects,
% and displaying the results.
obj = setupSystemObjects();
tracks = initializeTracks(); % Create an empty array of tracks.
nextId = 1; % ID of the next track
video = VideoWriter(write, 'MPEG-4');
video.FrameRate = obj.reader.FrameRate;
numFrames = obj.reader.NumFrames; 
% width = obj.reader.Width; height = obj.reader.Height;
frame_lines = repmat(struct,1,numFrames);
centrs = cell(numFrames, 1);
coefstr = cell(numFrames, 1);
% Detect moving objects, and track them across video frames.
open(video);
frame_count = 1;
while hasFrame(obj.reader)
    frame = readFrame(obj.reader);
    [out, bboxes, centroids, UnqLines, coef, pts, linepts] = improc();
    coefstr{frame_count} = coef;
    centrs{frame_count} = centroids;
    frame_lines(frame_count).unqlines = UnqLines;
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
function [out, bboxes, centroids, UnqLines, coef, pts, linepts] = improc()
% Finding sticker
Isub = imsubtract(frame(:,:,1), rgb2gray(frame));
Isub = imgaussfilt(Isub, 2);
Isub = imbinarize(Isub,0.2);
% Isub = bwareaopen(Isub, 300);
regprops = regionprops(Isub);
refbbox = regprops.BoundingBox;
linepts = [refbbox(1) refbbox(2)+refbbox(4) ... 
    refbbox(1)+refbbox(3) refbbox(2)];

    Img = rgb2gray(frame);
    % A set of filters
%     Img = imadjust(Img, [0.3 0.6]);
%     Img = imsharpen(Img, 'Amount',1.2);
%     Img = medfilt2(Img, [3 3]);
%     Img = imdiffusefilt(Img, 'NumberOfIterations', 10);
%     Img = imguidedfilter(Img);
%     Img = fibermetric(Img, 'ObjectPolarity', 'dark');
%     BW = imbinarize(Img, 0.2);
    Img = imgaussfilt(Img, 2);
    % Filtering out sticker region
    h = fspecial('average', 50);
    ov = 10;
    filtreg = roipoly(Img, [linepts(1)-ov linepts(1)-ov ...
        linepts(3)+ov linepts(3)+ov],...
        [linepts(2)+ov linepts(4)-ov linepts(4)-ov linepts(2)+ov]);
    Img = roifilt2(h,Img,filtreg);
    out = edge(Img, 'Canny', [0.001 0.2]);
%     out = bwareaopen(out, 30);
%     out = imclose(out, 50);
%     out2 = edge(rgb2gray(frame), 'Roberts');
    
    % Finding straigth lines using Hough transform
    [H,T,R] = hough(out, 'RhoResolution', 1);
    P  = houghpeaks(H, 1);
    UnqLines = houghlines(out, T, R, P, 'MinLength', 100);
    out = uint8(repmat(out, 1, 1, 3)) .* 255;
    
    % Calculating bounding boxes, centroids and straight line coefficients
    numUnqLines = length(UnqLines); mnan = [];
    bboxes = ones(numUnqLines, 4); centroids = ones(numUnqLines, 2);
    coef = ones(numUnqLines, 2); pts = ones(numUnqLines, 4);
    point1 = reshape([UnqLines.point1], 2, [])';
    point2 = reshape([UnqLines.point2], 2, [])';
    for m = 1:numUnqLines
        
        % Working with vertical lines that have Inf coefficient
        if point1(m, 1) == point2(m, 1)
            coef(m, :) = [NaN, point1(m, 1)]; mnan(end+1) = m;
        else
            coef(m, :) = polyfit([point1(m, 1), point2(m, 1)], ...
                [point1(m, 2), point2(m, 2)], 1);
        end
        % Finding corresponding pts on boundary from obtained coefficients
        pts(m, :) = lineToBorderPoints([coef(m, 1), -1, coef(m, 2)], ...
            size(Img));
        bboxes(m, :) = [min([pts(m, 1), pts(m, 3)]), ...
            max([pts(m, 2) pts(m, 4)]), ...
            abs(pts(m, 3) - pts(m, 1)), abs(pts(m, 4) - pts(m, 2))];
        centroids(m, :) = [min([pts(m, 1), pts(m, 3)]) + ...
            abs(pts(m, 3) - pts(m, 1))/2, ...
            min([pts(m, 2) pts(m, 4)]) + abs(pts(m, 2) - pts(m, 4))/2];
    end
    
    % Correcting bbox and centroids for vertical lines
    if ~isempty(mnan)
        for mn = mnan
            bboxes(mn, :) = [coef(mn, 2), size(Img, 2), ...
                0, size(Img, 2)];
            centroids(mn, :) = [coef(mn, 2), size(Img, 2)/2];
            pts(mn, :) = [coef(mn, 2) 0.5 coef(mn, 2) 720.5];
        end
    end
    
    % Filtering close to each other centroids
    tmpk = []; oldl = 2; newl = 1;
    while oldl > newl && size(centroids, 1) > 1
        oldl = size(centroids, 1);
        tmpv = centroids(newl, :);
        tmp = centroids(setdiff(1:end, newl), :);
        [~, dist] = dsearchn(tmp, tmpv);
        if dist > 10, newl = newl + 1; continue
        else, tmpk(end+1) = newl; centroids = tmp;
        end
    end 
    for t = tmpk     
        bboxes = bboxes(setdiff(1:end, t), :);
        pts = pts(setdiff(1:end, t), :);
    end  
    
    % Creating visual markers for mask video
    out = insertShape(out, 'Line', [point1 point2], ...
                'LineWidth', 3, 'Color', 'green', 'SmoothEdges', false);
    out = insertMarker(out, centroids, 'o', 'Size', 10, ...
                'Color', 'red');
    out = insertShape(out, 'Line', linepts, ...
                'LineWidth', 3, 'Color', 'green', 'SmoothEdges', false);
%     Isub = imsubtract(out(:,:,2), rgb2gray(out));
%     mask = imbinarize(Isub);
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
        'consecutiveInvisibleCount', {}, ...
        'endpts', {});
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
           predictedCentroid(2) + bbox(4) / 2, bbox(3:4)];
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
%             tracks(i).mvmnt = cost(i, :);
        end
        
        % Solve the assignment problem.
        costOfNonAssignment = 70;
        [assignments, unassignedTracks, unassignedDetections] = ...
            assignDetectionsToTracks(cost, costOfNonAssignment, 100);
 end
%% Updating tracks
function updateAssignedTracks()
    numAssignedTracks = size(assignments, 1);
    for i = 1:numAssignedTracks
        trackIdx = assignments(i, 1);
        detectionIdx = assignments(i, 2);
        centroid = centroids(detectionIdx, :);
        bbox = bboxes(detectionIdx, :);
        pt = pts(detectionIdx, :);

        % Correct the estimate of the object's location
        % using the new detection.
        correct(tracks(trackIdx).kalmanFilter, centroid);
        
        % Replace predicted bounding box with detected
        % bounding box.
        tracks(trackIdx).bbox = bbox;
        tracks(trackIdx).endpts = pt;

        
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
        if isempty(tracks), return;
        end
        invisibleForTooLong = 30;
        ageThreshold = 10;

        % Compute the fraction of the track's age for which it was visible.
        ages = [tracks(:).age];
        totalVisibleCounts = [tracks(:).totalVisibleCount];
        visibility = totalVisibleCounts ./ ages;

        % Find the indices of 'lost' tracks.
        lostInds = (ages < ageThreshold & visibility < 0.5) | ...
            [tracks(:).consecutiveInvisibleCount] >= invisibleForTooLong;

        % Delete lost tracks.
        tracks = tracks(~lostInds);
end
%%
function createNewTracks()
    centroids = centroids(unassignedDetections, :);
    bboxes = bboxes(unassignedDetections, :);
    endpts = pts(unassignedDetections, :);
    
    for i = 1:size(centroids, 1)
        centroid = centroids(i, :); bbox = bboxes(i, :);
        pt = endpts(i, :);
        
        % Create a Kalman filter object.
        kalmanFilter = configureKalmanFilter('ConstantAcceleration',...
            centroid, [5e1 1 1], [5e1, 1 1], 1*1e1);
        
        % Create a new track.
        newTrack = struct(...
            'id', nextId, ...
            'bbox', bbox, ...
            'kalmanFilter', kalmanFilter, ...
            'age', 1, ...
            'totalVisibleCount', 1, ...
            'consecutiveInvisibleCount', 0, ...
            'endpts', pt);
        
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
    frame = insertShape(frame, 'Line', pts, 'LineWidth', 5, ...
                'Color', 'red', 'SmoothEdges', false);
%     mask = uint8(repmat(mask, [1, 1, 3])) .* 255;
%     Img = uint8(repmat(Img, [1, 1, 3]));
    minVisibleCount = 70;
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
            relpts = cat(1, reliableTracks.endpts);
            numRelTracks = size(relpts, 1);
%             slope = zeros(1, numRelTracks);
            angle = zeros(1, numRelTracks);
            position = ones(numRelTracks, 2);
            stckang = atand((linepts(3)-linepts(1))./...
                (linepts(4)-linepts(2)))
            for i = 1:numRelTracks
                slope = (relpts(i, 3) - relpts(i, 1))./ ...
                    (relpts(i, 2) - relpts(i, 4));
                angle(i) = atand(slope);
                position(i, :) = [1, 200+60*(i-1)];
            end
            angle = angle + stckang;
            angle = cellstr(int2str(angle'));
            
            % Get ids.
            ids = int32([reliableTracks(:).id]);
            
            % Create labels for objects indicating the ones for
            % which we display the predicted rather than the actual
            % location.
            labels = cellstr(int2str(ids'));
            predictedTrackInds = ...
                [reliableTracks(:).consecutiveInvisibleCount] > 30;
            isPredicted = cell(size(labels));
            isPredicted(predictedTrackInds) = {' predicted'};
            labels = strcat(labels, isPredicted);
            anglelabels = strcat(labels, ': ', angle, ' degree');
            
            % Draw the objects on the frame.
            frame = insertObjectAnnotation(frame, 'circle', ...
                [bboxes(:, 1)+bboxes(:, 3)/2, ...
                bboxes(:, 2)-bboxes(:, 4)/2, ...
                5*ones(size(bboxes, 1), 1)], labels);
            frame = insertText(frame, position, anglelabels, ...
                'FontSize', 30, 'BoxColor', 'green');
            frame = insertShape(frame, 'Line', linepts, ...
                'LineWidth', 3, 'Color', 'green', 'SmoothEdges', false);
        end
    end
    
    % Display the mask and the frame.
    obj.maskPlayer.step(out); obj.videoPlayer.step(frame);
end
end