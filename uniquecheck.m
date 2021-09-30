if exist('newlines') ~= 1
    load uniquecheck 
end
findEqMat = [newlines.theta; newlines.rho]';
    [u, I, J] = unique(findEqMat, 'rows', 'first');
    hasDuplicates = size(u, 1) < size(findEqMat, 1);
    if hasDuplicates
        ixDupRows = setdiff(1:size(findEqMat,1), I);
        dupRowValues = unique(findEqMat(ixDupRows, :), 'rows');
        sameInd = cell(1, size(dupRowValues, 1));
        for i = 1:size(dupRowValues, 1)
            [tmp ~] = find(findEqMat == dupRowValues(i,:));
            sameInd{1, i} = unique(tmp');
        end
            uniqInd = setdiff(1:length(newlines), unique(cell2mat(sameInd)));
            UnqLines = newlines(1, uniqInd);
            store1 = []; store2 = [];
        for j=1:length(sameInd)
            for k = sameInd{j}
                store1 = [store1 newlines(k).point1(1)];
                store2 = [store2 newlines(k).point2(1)];
            end
            [~, index1] = min(store1);
            [~, index2] = max(store2);
            newlines(sameInd{j}(index2)).point1 = ...
                newlines(sameInd{j}(index1)).point1;
            UnqLines = [UnqLines newlines(sameInd{j}(index2))];
            
        end
    end