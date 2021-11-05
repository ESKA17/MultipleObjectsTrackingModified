function MotionBasedMultiObject()
% Create System objects used for reading video, detecting moving objects,
% and displaying the results.
close all; imaqreset
vid = videoinput('winvideo');
triggerconfig(vid, 'manual'); set(vid, 'TriggerRepeat', inf);
vid.FrameGrabInterval = 1;
src = getselectedsource(vid); frameRates = set(src, 'FrameRate');
fps = frameRates{1}; src.FrameRate = fps;
tracks = initializeTracks(); % Create an empty array of tracks.
nextId = 1; % ID of the next track
% video.FrameRate = obj.reader.FrameRate;
% numFrames = obj.reader.NumFrames; 
% width = obj.reader.Width; height = obj.reader.Height;
numFrames = 10000;
% Detect moving objects, and track them across video frames.
start(vid); frame_count = 1;
warning('off')
figure('Name','Video capture', 'NumberTitle','off')
while frame_count < numFrames + 1
    trigger(vid); frame = getsnapshot(vid);
    [bboxes, centroids, pts, linepts, myang, crns, refbbox] = improc();
    predictNewLocationsOfTracks();
    [assignments, unassignedTracks, unassignedDetections] = ...
        detectionToTrackAssignment();
    updateAssignedTracks();
    updateUnassignedTracks();
    deleteLostTracks();
    createNewTracks();
    frame = displayTrackingResults(frame);
    imshow(frame)
    frame_count = frame_count + 1; 
    flushdata(vid);
end
stop(vid);  delete(vid);

%% Image processing
function [bboxes, centroids, pts, linepts, myang, crns, refbbox, Isub] = improc()
    
% Finding a sticker
Isub = imsubtract(frame(:,:,1), rgb2gray(frame)); % red one
Isub = imgaussfilt(Isub, 2);
Isub = imbinarize(Isub);
Isub = bwareaopen(Isub, 300);
regprops = regionprops(Isub, 'Orientation', 'BoundingBox', 'Area');
areaFromStruct = [regprops.Area];
g = areaFromStruct < 750;
regprops = regprops(g);
linepts = [0 0 0 0]; myang = 0;
if ~isempty(regprops)
%     linepts = [refbbox(1) refbbox(2)+refbbox(4) ...
%         refbbox(1)+refbbox(3) refbbox(2)];
%     myang = regprops.Orientation;
%     Isubcrp = imcrop(Isub, refbbox);
    refbbox = extractfield(regprops, 'BoundingBox');
    refbbox = reshape(refbbox, [4, size(regprops, 1)])';
    % Cut out
    Isubcrp = cell([1, size(regprops, 1)]);
    crns = cell([1, size(regprops, 1)]);
    for i = 1:size(regprops, 1)
        Isubcrp{i} = imcrop(Isub, refbbox(i, :));
        crns{i} = pgonCorners(Isubcrp{i}, 5);
    end
    %Examine each one
    for ii = 1:size(regprops, 1)    
        if size(crns{ii}, 1) == 5
            angle = zeros(5, 1);
            ctmp = cat(1, crns{ii}, crns{ii}(1:2, :));
            for c = 1:5
                % Normalized vectors
                n1 = (ctmp(c+2,:)-ctmp(c,:)) / ...
                    norm(ctmp(c+2,:)-ctmp(c,:));
                n2 = (ctmp(c+1,:) - ctmp(c,:)) / ...
                    norm(ctmp(c+1,:) - ctmp(c,:));
                angle(c) = rad2deg(atan2(norm(det([n2; n1])),dot(n1, n2)));
%  angle(c) = rad2deg(atan2(abs((ctmp(c+1,1)-ctmp(c,1))*(ctmp(c+2,2)-ctmp(c,2))-(ctmp(c+2,1)-ctmp(c,1))*(ctmp(c+1,2)-ctmp(c,2))), ...
%                 (ctmp(c+1,1)-ctmp(c,1))*(ctmp(c+2,1)-ctmp(c,1))+(ctmp(c+1,2)-ctmp(c,2))*(ctmp(c+2,2)-ctmp(c,2))));
            end
            angle
            str = find(angle < 95 & angle > 85);
            shr = find(angle < 50);
            if length(str) == 2 && length(shr) == 1
                linepts = ...
                    [abs(ctmp(str(1)+1,1)-ctmp(str(2)+1,1))/2+...
                    min(ctmp(str(1)+1,1)-ctmp(str(2)+1,1)), ...
                    abs(ctmp(str(1)+1,2)-ctmp(str(2)+1,2))/2 + ...
                    min(ctmp(str(1)+1,2)-ctmp(str(2)+1,2)), ...
                    ctmp{ii}(shr + 1, 1), ctmp(shr + 1, 2)];
                slope = (linepts(3) - linepts(1))./ ...
                    (linepts(2) - linepts(4));
                myang = atand(slope)
                crns{ii}(:, 1) = ctmp(:, 1)+ refbbox(ii, 1); 
                crns{ii}(:, 2) = ctmp(:, 2)+refbbox(ii, 2)-refbbox(ii, 4); 
                refbbox = refbbox(ii, :);
            end
        end
    end
end
chck1 = exist('crns', 'var');
chck2 = exist('refbbox', 'var');
if  chck1 ~= 1 || chck2 ~=1
    crns = []; refbbox = [];
end
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
    Img = roifilt2(h, Img, filtreg);
    out = edge(Img, 'Canny', [0.001 0.2]);
%     out = bwareaopen(out, 30);
%     out = imclose(out, 50);
    
    % Finding straigth lines using Hough transform
    [H,T,R] = hough(out, 'RhoResolution', 1);
    P  = houghpeaks(H, 1);
    UnqLines = houghlines(out, T, R, P, 'MinLength', 100);
%     out = uint8(repmat(out, 1, 1, 3)) .* 255;
    
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
    
    % Filtering out close to each other centroids
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
%     out = insertShape(out, 'Line', [point1 point2], ...
%                 'LineWidth', 3, 'Color', 'green', 'SmoothEdges', false);
%     out = insertMarker(out, centroids, 'o', 'Size', 10, ...
%                 'Color', 'red');
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
function frame = displayTrackingResults(frame)
    % Convert the frame and the mask to uint8 RGB.
    frame = im2uint8(frame);
    frame = insertShape(frame, 'Line', pts, 'LineWidth', 5, ...
                'Color', 'red', 'SmoothEdges', false);
%     mask = uint8(repmat(mask, [1, 1, 3])) .* 255;
%     Img = uint8(repmat(Img, [1, 1, 3]));
    minVisibleCount = 30;
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
            for i = 1:numRelTracks
                slope = (relpts(i, 3) - relpts(i, 1))./ ...
                    (relpts(i, 2) - relpts(i, 4));
                angle(i) = atand(slope);
                position(i, :) = [1, 200+60*(i-1)];
            end
            if ~isequal(linepts, [0, 0, 0, 0])
            angle = angle + myang - 90;
            frame = insertShape(frame, 'Line', linepts, ...
                'LineWidth', 3, 'Color', 'green', 'SmoothEdges', false);
            frame = insertShape(frame, 'Rectangle', refbbox, ...
                'LineWidth', 3, 'Color', 'green', 'SmoothEdges', false);
            end
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
            if size(crns, 1)>3 && ~isempty(crns)
                v = crns(:, 2); crns(:, 2) = crns(:, 1); crns(:, 1) = v;
                frame = insertShape(frame, 'Polygon', ...
                    reshape(crns.',1,[]));
            end
            frame = insertText(frame, position, anglelabels, ...
                'FontSize', 30, 'BoxColor', 'green');
        end
    end
    
end
end