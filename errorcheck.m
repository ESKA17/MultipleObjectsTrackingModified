clear
load frame
I = rgb2gray(frame);
BW = imbinarize(I);
out = edge(I, 'Roberts');
[H,T,R] = hough(out);
P  = houghpeaks(H, 3, 'threshold', ceil(0.3*max(H(:))));
lines = houghlines(BW, T, R, P, 'FillGap', 5, 'MinLength', 7);
out = uint8(repmat(out, 1, 1, 3)) .* 255;
    for k = 1:length(lines)
        xy = [lines(k).point1; lines(k).point2];
        xspace = xy(1, 1):xy(2, 1);
        yspace = fix(xy(1, 2) + xspace*(xy(2,2)-xy(1, 2))/(xy(2, 1)-xy(1, 1)));
        if sum(yspace < 0) > 0
            yspace = yspace(yspace > 0);
            xspace = xspace(1:length(yspace));
        end
% error check block
        try
        for i = 1:length(xspace)
            out(yspace(i), xspace(i),1) = 0;
            out(yspace(i), xspace(i),3) = 0;
        end
        catch
            tmpxSpc = xspace;
            tmpySpc = yspace;
            tmpki = [k i];
            return
        end
        
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
    mask = zeros(size(BW));
    mask(find(out(:, :, 2)> 200 & out(:, :, 1) == 0 & ...
        out(:, :, 1) == 0)) = 1;