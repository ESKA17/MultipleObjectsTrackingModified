obj.reader = VideoReader('example1.mp4');
frame = readFrame(obj.reader);
I = rgb2gray(frame);
BW = imbinarize(I);
out = edge(I, 'Roberts');
[H,T,R] = hough(out);
P  = houghpeaks(H, 3, 'threshold', ceil(0.3*max(H(:))));
lines = houghlines(BW, T, R, P, 'FillGap', 5, 'MinLength', 7);
out = uint8(repmat(~out, 1, 1, 3)) .* 255;
    for k = 1:length(lines)
        xy = [lines(k).point1; lines(k).point2];
        xspace = xy(1, 2):xy(2, 2);
        yspace = fix(xy(1, 2) + xspace*(xy(2,2)-xy(1, 2))/(xy(2, 1)-xy(1, 1)));
        for i = 1:length(xspace)
            out(xspace(i), yspace(i),1) = 0;
            out(xspace(i), yspace(i),3) = 0;
        end
        centroids(k, :) = [xy(1, 1) + (xy(2, 1) - xy(1, 1))/2, ...
            xy(1, 2) + (xy(2, 2) - xy(1, 2))/2];
        bboxes(k, :) = [xy(1, :) (xy(2, 1) - xy(1, 1)) xy(2, 2) - xy(1, 2)];
    end
    
    mask = out(:, :, 2) > 200;