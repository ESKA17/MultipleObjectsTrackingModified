v = VideoReader('example1.mp4');
lines = struct;
for i = 1:v.NumFrames
frame = read(v, i);
I = rgb2gray(frame);
BW = imbinarize(I);
out = edge(I, 'Roberts');
% out = imclose(out, strel('arbitrary', [20 20]));
% out = imopen(out, strel('arbitrary', [10 10]));
% out = imfill(out, 'holes');
[H,T,R] = hough(out);
P  = houghpeaks(H, 3, 'threshold', ceil(0.3*max(H(:))));
lines = houghlines(BW, T, R, P, 'FillGap', 5, 'MinLength', 7);
imshow(out), hold on
    for k = 1:length(lines)
        xy = [lines(k).point1; lines(k).point2];
        plot(xy(:,1), xy(:,2), 'LineWidth', 2, 'Color','green');            
    end
    hold off
F(i) = getframe(gcf);                       
end

video = VideoWriter('myvideo.avi'); 
video.FrameRate = v.FrameRate;
open(video);
for i=1:length(F)
    frame = F(i) ;    
    writeVideo(video, frame);
end
close(video);