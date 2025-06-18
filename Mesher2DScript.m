
% this is the script file for the 2D mesher program
% it needs fileIO which is the file reader for the program and others like
% it
% check the DATA file and the README file for the input. THere is a ppt
% show that explains how this works 
% available comands the first three lines are self explanatory

% try this with the file multisegment test in the DATA folder to see what
% kind of mesh is generated. 

% HOW IT WORKS
% it creates a set of points that form an equilateral triangular grid.
% then it slighlty scrambles the interior points by adding a bit of noise
% then a delaunay triangulation is created
% then the shape of the triangles are adjusted and transions between coarse
% and dense grids are smoothened by a couple of different smoothening
% methods 

close all
M=Mesher2D;
FileIO.readFile(M);
M.createElements(0.65);
    % this is the one that createsthe points and the elements. It is controlled
    % by a single input variable "temperature", that is between 0 and 1  that sets the degree of
    % randomness. 0 will give you a mostl y structured mesh. As you increase
    % the value, the grid will become more random and also more "clumpy" so I
    % recommend  around .5 o to about 0.7. 
M.adjustTriangulation(10);
    % this one makes the triangles "more equilateral" by penalizing edge length
    % differrences nomralized by average edge legnth. takes one input which
    % is the number of epochs to try.
M.smoothGraph(10);
    % smooth graph is a laplacan smoothing operation that moves every point to
    % the centroid of its neigbhors, it does slighlyt adjust
M.spreadPoints(500);
    % you can slightly improve the mesh by repelling the points a bit sow
    % crowded regions will push the points to less dense regions. 
M.showMeshQuality;
    % This displays the mesh and  computes and shows various quality measures on the mesh. 
FileIO.saveMyDataToFile(M.coords,M.elements);
    % this saves to a file specifice by you and stores only nodal
    % coorditants and elements. 

          