RGB = imread('example.jpg');
I = rgb2gray(RGB);
BW = imbinarize(I);
rotI = imrotate(BW,33,'crop');
out = edge(I, 'Prewitt');
% imshow(out)


[H,T,R] = hough(out);
P  = houghpeaks(H, 1,'threshold',ceil(0.3*max(H(:))));
x = T(P(:,2)); y = R(P(:,1));

lines = houghlines(BW,T,R,P,'FillGap',5,'MinLength',7);
figure, imshow(out), hold on
max_len = 0;
for k = 1:length(lines)
   xy = [lines(k).point1; lines(k).point2];
   plot(xy(:,1),xy(:,2),'LineWidth',2,'Color','green');

   % Plot beginnings and ends of lines
   plot(xy(1,1),xy(1,2),'x','LineWidth',2,'Color','yellow');
   plot(xy(2,1),xy(2,2),'x','LineWidth',2,'Color','red');

   % Determine the endpoints of the longest line segment
   len = norm(lines(k).point1 - lines(k).point2);
   if ( len > max_len)
      max_len = len;
      xy_long = xy;
   end
end