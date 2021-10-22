%% Draft code
% Finding straigth lines using Hough transform
    [H,T,R] = hough(out, 'RhoResolution', 1);
%     P  = houghpeaks(H, 3, 'threshold', ceil(0.3*max(H(:))));
%     lines = houghlines(BW, T, R, P, 'FillGap', 5, 'MinLength', 7);
    P  = houghpeaks(H, 1);
    UnqLines = houghlines(out, T, R, P, 'MinLength', 100);
    out = uint8(repmat(out, 1, 1, 3)) .* 255;
    
%     % Filtering low area bounding boxes
%     numLin = length(UnqLines); missInd = []; indArray = 1:numLin;
%     bbtmp = ones(numLin, 4);
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
%     to be based on coefficients rather than output from houghlines, that
%     was implemented in early revisions
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
    
    % Calculating bounding boxes, centroids and straight line coefficients
    numUnqLines = length(UnqLines); mnan = [];
    bboxes = ones(numUnqLines, 4); centroids = ones(numUnqLines, 2);
    coef = ones(numUnqLines, 2); pts = ones(numUnqLines, 4);
    point1 = reshape([UnqLines.point1], 2, [])';
    point2 = reshape([UnqLines.point2], 2, [])';