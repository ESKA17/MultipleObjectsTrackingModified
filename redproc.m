readfrom = 'alpha2.mp4'; writeto = 'test2';
reddetection(readfrom, writeto);
function UnqLines = reddetection(readfrom, writeto)
obj = setupSystemObjects(readfrom);
video = VideoWriter(writeto, 'MPEG-4');
open(video);

while hasFrame(obj.reader)
frame = readFrame(obj.reader);
I = frame;
Isub = imsubtract(I(:,:,1), rgb2gray(I));
Isub = imgaussfilt(Isub, 2);
Isub = imbinarize(Isub,0.2);
% Isub = bwareaopen(Isub, 300);
regprops = regionprops(Isub);

Isub = uint8(repmat(Isub, 1, 1, 3));
Isub(Isub == 0) = 255;
Isub(Isub(:, :, 1) == 1) = 255;
% Isub = insertShape(Isub, 'Rectangle', regprops.BoundingBox, ...
%                 'LineWidth', 3, 'Color', 'green', 'SmoothEdges', false);
bbox = regprops.BoundingBox;
linepts = [bbox(1) bbox(2)+bbox(4) bbox(1)+bbox(3) bbox(2)];
Isub = insertShape(Isub, 'Line', linepts, ...
                'LineWidth', 3, 'Color', 'green', 'SmoothEdges', false);
writeVideo(video, Isub);
obj.maskPlayer.step(Isub); obj.videoPlayer.step(frame);
end
close(video);
end

function obj = setupSystemObjects(readfrom)
     % Create a video reader.
     obj.reader = VideoReader(readfrom);
     
     % Create two video players, one to display the video,
     % and one to display the foreground mask.
     obj.maskPlayer = vision.VideoPlayer('Position', [740, 400, 700, 400]);
     obj.videoPlayer = vision.VideoPlayer('Position', [20, 400, 700, 400]);
end


