function MotionBasedMultiObjectTrackingModified()
% Create System objects used for reading video, detecting moving objects,
% and displaying the results.
obj = setupSystemObjects();

tracks = initializeTracks(); % Create an empty array of tracks.

nextId = 1; % ID of the next track

video = VideoWriter('myvideo2.avi');
video.FrameRate = obj.reader.FrameRate;

% Detect moving objects, and track them across video frames.
open(video);
while hasFrame(obj.reader)
frame = readFrame(obj.reader);
I = rgb2gray(frame);
BW = imbinarize(I);
out = edge(I, 'Roberts');
[H,T,R] = hough(out);
P  = houghpeaks(H, 3, 'threshold', ceil(0.3*max(H(:))));
lines = houghlines(BW, T, R, P, 'FillGap', 5, 'MinLength', 7);
out = uint8(repmat(out, 1, 1, 3)) .* 255;
img = out;
hFig = figure('Name','APP',...
    'Numbertitle','off', 'Units', 'normalized',...
    'Position', [0 0 1 1], 'MenuBar', 'none', 'Toolbar', 'none', ...
    'WindowStyle', 'modal', 'WindowState', 'fullscreen',...
    'Color',[0.5 0.5 0.5], 'OuterPosition', [0 0 1 1], ...
    'InnerPosition', [0 0 1 1]);
fpos = get(hFig,'Position');
axOffset = (fpos(3:4)-[size(img,2) size(img,1)])/2;
% f = figure('Visible','off', 'Units','normalized',  'MenuBar', 'None',...
%     'ToolBar', 'None',...
%     'WindowState', 'fullscreen', 'OuterPosition',[0 0 1 1], ...
%     'Papersize', [1 1]);
% set(f,'Renderer', 'ZBuffer')
% set(gca, 'Visible', 'off')
ha = axes('Parent', hFig, 'Units','normalized',...
            'Position', [0 0 1 1], 'OuterPosition', [0 0 1 1]);
myImg = imshow(out,'Parent',ha);
hold on

    for k = 1:length(lines)
        xy = [lines(k).point1; lines(k).point2];
        plot(xy(:,1), xy(:,2), 'LineWidth', 2, 'Color','green');
%         set(gca, 'Visible', 'off')
   

%         xspace = xy(1, 1):xy(2, 1);
%         yspace = fix(xy(1, 2) + xspace*(xy(2,2)-xy(1, 2))/(xy(2, 1)-xy(1, 1)));
%         if sum(yspace < 0) > 0
%             yspace = yspace(yspace > 0);
%             xspace = xspace(1:length(yspace));
%         end
%          for i = 1:length(xspace)
%             out(yspace(i), xspace(i),1) = 0;
%             out(yspace(i), xspace(i),3) = 0;
%         end
         if xy(1, 2) <= xy(2, 2)
        centroids(k, :) = [xy(1, 1) + (xy(2, 1) - xy(1, 1))/2, ...
            xy(1, 2) + (xy(2, 2) - xy(1, 2))/2];
        bboxes(k, :) = [xy(1, 1) xy(2, 2) (xy(2, 1) - xy(1, 1))...
            xy(2, 2) - xy(1, 2)];
        else 
        centroids(k, :) = [xy(1, 1) + (xy(2, 1) - xy(1, 1))/2, ...
            xy(2, 2) + (xy(1, 2) - xy(2, 2))/2];
        bboxes(k, :) = [xy(1, 1) xy(1, 2) (xy(2, 1) - xy(1, 1))...
            xy(1, 2) - xy(2, 2)];
        end
    end
%     frame = out;
        hold off
        frame = getframe(gcf).cdata;
        clf 
        close
    Isub = imsubtract(frame(:,:,2), rgb2gray(frame));
    mask = imbinarize(Isub);
%     stats = regionprops(mask, 'Centroid', 'BoundingBox');
%     centroids = cat(1, stats.Centroid);
%     bboxes = cat(1, stats.BoundingBox);


    predictNewLocationsOfTracks();
    [assignments, unassignedTracks, unassignedDetections] = ...
        detectionToTrackAssignment();

    updateAssignedTracks();
    updateUnassignedTracks();
    deleteLostTracks();
    createNewTracks();

    displayTrackingResults();
    writeVideo(video, frame);
end
close(video);

%%
 function obj = setupSystemObjects()
        % Initialize Video I/O
        % Create objects for reading a video from a file, drawing the tracked
        % objects in each frame, and playing the video.

        % Create a video reader.
        obj.reader = VideoReader('example1.mp4');

        % Create two video players, one to display the video,
        % and one to display the foreground mask.
        obj.maskPlayer = vision.VideoPlayer('Position', [740, 400, 700, 400]);
        obj.videoPlayer = vision.VideoPlayer('Position', [20, 400, 700, 400]);
        

 end
%%
function tracks = initializeTracks()
        % create an empty array of tracks
        tracks = struct(...
            'id', {}, ...
            'bbox', {}, ...
            'kalmanFilter', {}, ...
            'age', {}, ...
            'totalVisibleCount', {}, ...
            'consecutiveInvisibleCount', {});
end
%%
function predictNewLocationsOfTracks()
        for i = 1:length(tracks)
            bbox = tracks(i).bbox;

            % Predict the current location of the track.
            predictedCentroid = predict(tracks(i).kalmanFilter);

            % Shift the bounding box so that its center is at
            % the predicted location.
            predictedCentroid = predictedCentroid - bbox(3:4) / 2;
            tracks(i).bbox = [predictedCentroid, bbox(3:4)];
        end
end
%%
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
        costOfNonAssignment = 20;
        [assignments, unassignedTracks, unassignedDetections] = ...
            assignDetectionsToTracks(cost, costOfNonAssignment);
 end
%%
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
%%
function updateUnassignedTracks()
        for i = 1:length(unassignedTracks)
            ind = unassignedTracks(i);
            tracks(ind).age = tracks(ind).age + 1;
            tracks(ind).consecutiveInvisibleCount = ...
                tracks(ind).consecutiveInvisibleCount + 1;
        end
end
%%
function deleteLostTracks()
        if isempty(tracks)
            return;
        end

        invisibleForTooLong = 20;
        ageThreshold = 10;

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
            kalmanFilter = configureKalmanFilter('ConstantVelocity', ...
                centroid, [200, 50], [100, 25], 25);

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

        minVisibleCount = 5;
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
                    [reliableTracks(:).consecutiveInvisibleCount] > 0;
                isPredicted = cell(size(labels));
                isPredicted(predictedTrackInds) = {' predicted'};
                labels = strcat(labels, isPredicted);

                % Draw the objects on the frame.
                frame = insertObjectAnnotation(frame, 'circle', ...
                    [bboxes(:, 1)-bboxes(:, 3)/2 bboxes(:, 2)-bboxes(:, 4)/2 ...
                    10*ones(size(bboxes, 1), 1)], labels);

                % Draw the objects on the mask.
                mask = insertObjectAnnotation(mask, 'circle', ...
                    [bboxes(:, 1)-bboxes(:, 3)/2 bboxes(:, 2)-bboxes(:, 4)/2 ...
                    10*ones(size(bboxes, 1), 1)], labels);
            end
        end

        % Display the mask and the frame.
        obj.maskPlayer.step(mask);
        obj.videoPlayer.step(frame);
end

end