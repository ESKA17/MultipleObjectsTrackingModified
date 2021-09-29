figure (1)
clf
hold on
for k = 1:length(lines)
        xy = [lines(k).point1; lines(k).point2];
        plot(xy(:, 1), xy(:,2), 'color',rand(1,3), 'LineWidth', 1.5)
end
hold off
        