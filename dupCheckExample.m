x = [
    1 1
    2 2
    3 3
    4 4
    2 2
    3 3
    3 3
    ];
[u,I,J] = unique(x, 'rows', 'first');
hasDuplicates = size(u,1) < size(x,1);
ixDupRows = setdiff(1:size(x,1), I);
dupRowValues = unique(x(ixDupRows,:), 'rows');
myInd = cell(1, size(dupRowValues, 1));
for i = 1:size(dupRowValues, 1)
    [tmp ~] = find(x == dupRowValues(i,:));
    myInd{1, i} = unique(tmp');
end
