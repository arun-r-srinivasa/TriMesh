classdef Mesher2D<handle
    %MESHMAKER CLASS : A method for creating unstructured two dimensional triangular meshes 
    %
    %   Given an input file that describes (a) the boundary of the object
    %   using some known curves (polygons, arcs of circles and splines (b)
    %   a refinement region that sets how fine the mesh needs to be
    % the input files follow the convention for all input files (2 examples
    % are triangle_hole and triangle_hole_refined.txt
    %
    %  The mesh will have a few small triangles since it is based on a rand
    %  seed generator that will occassionally create clumps. 
    %
    %   Avaliable public functions 
    %       M=MeshMaker2D;
    %       M.createPoints(temperatue );degree of
    %       randomness of the mesh. Typical value is about 0.7. 0 will give
    %       a mostly equilateral triangular mesh
    %       M. smoothTriangulation (n_cycles). it sets the number of cycles
    %       to try. 
    %      This will slighlty change the mesh and remove very small angles 
    %       M.showMeshQuality; shows histogram of the smallest mesh angles  

    properties
        
        coords; % noadal coordinates 
        elements;

        % triangulation  properties
        tri_edges; % E by 2 list of uniuqe edges of the triangulation 
        element_edges; %3*M% the list of edge numbers corresponding to each element 
        degree;% degree of each node objelement_edges
        adj; %adjacencey matrix for the mesh graph 

        % boundary properties 
        loop_count=1;
        loops={};  
        n_bdry_nodes=0;% the first k nodes are the boundary nodes
        xyb; % these are the boundary nodee each loop is separated by NaN so that we can use it to find out if points are inside or outside 
        bdry_edges;% list of edges that form the boundary 
        
        % density refinement information 
        regions={}; % these are exactly like the l0ops but are meant fo refinement
        R0;
        UIFig;
        original_fig;
        new_fig;
        angle_fig;
        length_fig;
        radius_fig;
        ratio_fig;
       
    end

    methods
        

%^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
%  INPUT PARSING FUNCTIONS: Needs to work with FileReader
%^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
%$$$$$$$$$$$$$$$$$PRIVATE FUNCTIONS $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$ 
%--------------------------------------------------------------------------
        function setLOOP(obj,lines)
            % defines the boundary of the region that we are interested in
            %TYPICAL INPUT LINE
            % % example of a line
            %*LOOP <---- a new loop begins
            % 1 <----- loop is made up of only 1 segment
            %POLYGON,0.1, -2,-2, 2,-2, 0,2, -2,-2<--- defninntion of a
            %segment. first data is segment type (POLYGON) 
            % next noe is the separation between points (0.1 units)
            % Note that the last point is the same as the first
            %indicating closed loop
            % you can add segments to the loop
            N=numel(lines);
            validOpts = ["POLYLINE" "CIRCLE" "SPLINE"]; 
            for i=1:N
                a=split(lines{i},',');
                curve_name=a{1};% first data is the type of curve
                curve_name  = validatestring(curve_name,validOpts,mfilename,"curve_name");
                obj.loops{obj.loop_count}.curve_name{i}=curve_name;
                v=(str2double(a(2:end)));
                if(any(isnan(v)))
                    error("parseNumericStrings:NonNumericInput", ...
              "some  element(s) of "+ curve_name+" in LOOP are not numeric: %s", lines{i});
                end
                obj.loops{obj.loop_count}.segment_params{i}=v';% this contains the information about the curve
            end
            obj.loop_count=obj.loop_count+1;
        end
 %-------------------------------------------------------------------------
        function setREFINEMENT_REGIONS(obj,lines)
            % this is a list of nested loops that define regions with
            % different mesh densities 
            N=numel(lines);
            validOpts = ["POLYLINE" "CIRCLE" "SPLINE"]; 
            for i=1:N
                a=split(lines{i},',');
                curve_name=a{1};
                curve_name  = validatestring(curve_name,validOpts,mfilename,"curve_name");
                b1=(str2double(a(2:end)))';
                if(any(isnan(b1)))
                    error("parseNumericStrings:NonNumericInput", ...
              "some  element(s) of "+ curve_name+" in REFINEMENT_REGIONS are not numeric: %s", lines{i});
                end
                obj.regions{i}.curve_name=curve_name;
                if(i==1 && b1(1)~=1)
                    error("parseNumericStrings:WrongInput",...
                        "first curve in the REFINEMENT_REGIONS must be level 1 (ie bounding box); i got : %s ", lines{1})
                end
                obj.regions{i}.depth=b1(1); % this is the subset nesting: 1= outer boundary, 2= subset of 1,  3 =subset of 2 
                obj.regions{i}.R0=0;  % thi is the size of the loop    
                obj.regions{i}.loop_params=b1(2:end); % information required to draw the loop 
                obj.regions{i}.xyr=[];
            end
        end
          function setBOUNDING_BOX(obj,lines)
              N=numel(lines);
              obj.xlims=str2num(lines{1});
              obj.ylims=str2num(lines{2});
              obj.fx=str2func(lines{3});
              obj.fy=str2func(lines{4});
              obj.n_pts=str2double(lines{5});
        
          end


 %-------------------------------------------------------------------------
 function createElements(obj,temperature)
        % creates points that are on a regular grid and then slighly
        % scrambles them based on the "tempeature" which should be a value
        % between 0 and 1
        % also trianglulates the point 
            obj.assignBoundaryPoints;
            figure
            scatter(obj.xyb(1,:),obj.xyb(2,:),10,'filled')
            hold on
            obj.assignInteriorPoints(temperature);
            scatter(obj.coords(:,1), obj.coords(:,2), 10, 'filled')
            axis equal
            hold off
            obj.triangulateNodes;
            obj.assignTriangulationProperties;
            obj.smoothGraph(500);
            obj.fixBoundaryElements; % adjusts points that are too close to the boundary 
            obj.triangulateNodes;
            obj.assignTriangulationProperties;
            obj.createFigs;
            obj.showTriangulation(obj.original_fig,'b')

        end


%^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
        
%--------------------------------------------------------------------------
         function assignBoundaryPoints(obj)
            n_loops=length(obj.loops);
            obj.xyb=[];
            % discretize eqach loop and segment. store this in the boundary
            % locations.
            R_list=[];
            %bdry_node_idx=zeros(n_loops,1);
            start=1;
            obj.bdry_edges=[];
            % for each loop
            %   for each segment that makes the loop
            %     fidn the curve type and the segement parameters
            %     compute the points on the segnement
            % add it to the list of boundary points
            % get average separration and add it to the list

            for i=1:n_loops
                a=obj.loops{i};
                max_segments=length(a.segment_params);
                n_nodes_in_loop=0;
                for j=1:max_segments
                    [xy,R0]=obj.createBoundaryPoints(a.curve_name{j},a.segment_params{j});
                     obj.xyb=[obj.xyb,xy(:,1:end-1)]; % 2 rows many columns 
                     obj.loops{i}.bdrynodes{j}=xy; 
                     R_list=[R_list,R0];
                     % keep track of now mapy points are there in the ith
                     % loop.
                     n_nodes_in_loop=n_nodes_in_loop+size(xy,2)-1; % dont double count the last node. 
                end
                obj.xyb=[obj.xyb,[NaN;NaN]]; % add NANS to separate the loops 
                fin=start+n_nodes_in_loop-1;
                edgesk=[start:fin-1;start+1:fin];% stagger the nodes to create edges
                obj.bdry_edges=[obj.bdry_edges;edgesk';fin,start];%connect the ends
                start=fin+1;
            end
            obj.R0=mean(R_list);
        end

%--------------------------------------------------------------------------
    
        function assignInteriorPoints(obj,temperature)
            % assigns interior points based on the required closeness and randomness 
            % it creates and equilateral triangular grid of a given density
            % for each loop
            % then removes the points that fall outside the boundary 
            % Make Region boundary for the purpose  of finding n the points
            % are inside. 
            n_regions=numel(obj.regions);
            for i=1:n_regions
                the_region=obj.regions{i};
                [xy,R0]=obj.createBoundaryPoints(the_region.curve_name,the_region.loop_params);
                obj.regions{i}.xyr=xy(:,1:end-1);
                obj.regions{i}.R0=R0;
            end

            xyin=[]; % these are the interior points 
            for i=1 : n_regions
                R=obj.regions{i}.R0;
                XYi=obj.regions{i}.xyr; % boundary of the region 
                % find the bounding box and the spacings 
                [x0,x1]=bounds(XYi,2);
                dx=max(x1-x0);

                % Now create an equlateral triangle mesh by moving
                % successive rows by a certaina amount 
                nx=ceil(dx/R);
                ny=ceil(dx/ ((sqrt(3)/2) * R));
                [X,Y]=meshgrid(0:nx, 0:ny); %Just a gid of integers
                x = R * ( X + 0.5 * mod(Y, 2) )+x0(1); % scaled by R and every alternate mod (Y,2)one is moved one 1/2 spacing 
                y=(sqrt(3)/2) * R * Y+x0(2);
                xy=[x(:),y(:)]'+ temperature*(rand(2,(nx+1)*(ny+1))-0.5)*R;% add a bit of randomnes to the regular triangle
                % trim points OUTSIDE  the current region 
                [in,on]=inpolygon(xy(1,:),xy(2,:),XYi(1,:),XYi(2,:));
                inside=logical(in.*(~on));

                % trim points that are inside any subset. 
                % subsets are regions whose depth is greater than the
                % current regions depth
                depthi=obj.regions{i}.depth;
                for j=1:n_regions
                    depthj=obj.regions{j}.depth;
                    % find points that are inside any subset. 
                    if(depthj>depthi)
                        XYj=obj.regions{j}.xyr;
                        in=inpolygon(xy(1,:),xy(2,:),XYj(1,:),XYj(2,:));
                        inside=logical(inside.*(~in)); % keep only the ones that are in the region but not in its subsets. 
                    end
                end
                xyin=[xyin,xy(:,inside)];
            end

            % finally trim all the points that are outside the actual deomain 
            [in,on]=inpolygon(xyin(1,:),xyin(2,:),obj.xyb(1,1:end-1),obj.xyb(2,1:end-1));
            inside=logical(in.*(~on));

            idx=find(~isnan(obj.xyb(1,:)));
            % list the coords with the first being the boundary coordinates
            obj.n_bdry_nodes=length(idx);
            obj.coords=[obj.xyb(1,idx)',obj.xyb(2,idx)';xyin(:,inside)'];
            obj.n_bdry_nodes=length(idx);
        end

        
 %-------------------------------------------------------------------------
        function triangulateNodes(obj)
            % this function retriangulates aany set of points that make up
            % the region. 
             [inAll,onAll]=inpolygon(obj.coords(:,1),obj.coords(:,2),obj.xyb(1,1:end-1),obj.xyb(2,1:end-1));
            obj.coords(~inAll,:)=[]; 
            DT=delaunayTriangulation(obj.coords,obj.bdry_edges); % constrained triangulation! forces the boundary edges to be part of the triangulation 
            % but it will create trainagles that are not entirely in the
            % region
            %^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
            % keep only the triangles that are inside the domain boundary
            
            % check of all the points are inside the region or on its
            % boundary
            tris=DT.ConnectivityList; % MX 3 these are the elements.
            xvals = obj.coords(tris, 1);   % 3M x1
            yvals = obj.coords(tris, 2);   % 3M x1 
            [inAll,onAll]=inpolygon(xvals,yvals,obj.xyb(1,1:end-1),obj.xyb(2,1:end-1));

            % next check that the centroids of the elemments are in
            % eliminates cases WITH CONCAVE BOUNDARIES where all three
            % points are on the boundary but the element is outside 
            xE=reshape(xvals,[],3);
            yE=reshape(yvals,[],3);
            xC=mean(xE,2);
            yC=mean(yE,2);
            [inC, onC]=inpolygon(xC,yC,obj.xyb(1,1:end-1),obj.xyb(2,1:end-1));

            % now keep only the elements that are entirely inside 
            inMatrix = reshape(inAll, [], 3);   % M x 3
            onMatrix = reshape(onAll, [], 3);   % M x 3
            triIsInside = all(inMatrix | onMatrix, 2) &(inC) &(~onC);
            obj.elements=tris(triIsInside,:);% this is the set that we are really interested 
            %^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
        end

        function assignTriangulationProperties(obj)
           tris=obj.elements;
            all_edges = [ 
                tris(:, [1,2]);
                tris(:, [2,3]);
                tris(:, [3,1])
            ];
            % this lists the edges in element sequence.
            % Sort each edge pair so that edge (i,j) always has i<j
            all_edges = sort(all_edges, 2);
            % Step 2: Get unique edges
            [obj.tri_edges, ~, obj.element_edges] = unique(all_edges, 'rows'); % what is this 
            % obj.element_edges is a row vector such that
            % obj.tri_edges(obj.element_edges) reproduces the original edge
            % element_edge(i) is the tri_ege that corresponds to all_edge # i; 
            % so you rsehape it as ([],3) you will get the tri_egege # for each edge of the element. 
            % list
            % the key idea here is that 
            i=obj.tri_edges(:,1);
            j=obj.tri_edges(:,2);
            n_nodes=size(obj.coords,1);
            obj.adj = sparse(i, j, 1, n_nodes, n_nodes);   % Only fills upper (or lower) triangular part
            obj.adj =  obj.adj +  obj.adj.';                 % Make it symmetric
            obj.degree=sum(obj.adj,2); 
        end

 %-------------------------------------------------------------------------M.nodeRs(
        function fixBoundaryElements(obj)
            % removes nodes that are too close to the boundary 
            % if the area of the elmement with one edge on the boundary is
            % less than 40% of the square of the boundary edge length, then
            % the element is too close 
             TR=triangulation(obj.elements, obj.coords);
             % find the elements to which the boundaries are attached; 
             bdry_elements=edgeAttachments(TR,obj.bdry_edges);
             removed_nodes=[];
            for i=1:numel(bdry_elements)
                j=bdry_elements{i};
                node_numbers=obj.elements(j,:);
                % find the nodes in the interior and the boundary 
                idx=find(node_numbers>obj.n_bdry_nodes);
                jdx=find(node_numbers<=obj.n_bdry_nodes);
                Rs=obj.coords(node_numbers,:);
                % find the area using the shoelace formaula
                area=0.5*abs(Rs(1,1)*(Rs(2,2)-Rs(3,2))+Rs(2,1)*(Rs(3,2)-Rs(1,2))+Rs(3,1)*(Rs(1,2)-Rs(2,2)));
                d=norm(Rs(jdx(1),:)-Rs(jdx(2),:));
                if(area/d^2<0.3) % remove nodes that form very "flat" triangles 
                    removed_nodes=[removed_nodes,node_numbers(idx)];
                end
            end
            % remove the nodes that are trapped inside other triangles 
            trapped_nodes=find(obj.degree<=4);
            inside=(trapped_nodes>obj.n_bdry_nodes);
            removed_nodes=[removed_nodes,trapped_nodes(inside)'];
            obj.coords(removed_nodes,:)=[];
            
        end
%--------------------------------------------------------------------------
        function movePts(obj,n_cycles,learn_rate,TOL,gradFn)
            for i=1:n_cycles
             gradient=obj.computeVectorizedGrad(gradFn);
             gradient(1:obj.n_bdry_nodes,:)=0; % dont move the boundary nodes;
             gradnorm=vecnorm(gradient,2,2);
             error=mean(gradnorm)*obj.R0;
             offset=1;
             if(error<TOL)
                    break;
            end
            obj.coords=obj.coords-learn_rate*obj.R0*gradient./(gradnorm+offset);
            end
        end

        function adjustTriangulation(obj,n_epochs)
            % uses a triangle adjusting agoritm to make the triangles more
            % "equilateral"
            learn_rate=0.1; 
            offset=0.01/obj.R0; % this is the treshold beyond which the gradient will be scaled down to 
            TOL=0.01;
            % this prevents elements from becoming too small 
            k=1000; %Repelling spring stiffness 
            for i= 1:n_epochs
                obj.movePts(50,learn_rate,TOL,@obj.computeTriangleGrad)
                obj.triangulateNodes;
                obj.assignTriangulationProperties;
            end
            obj.fixBoundaryElements;
            obj.triangulateNodes;
            obj.assignTriangulationProperties; 
        end
%--------------------------------------------------------------------------
         function spreadPoints(obj,n_cycles)
             
             obj.movePts(n_cycles,0.5,0.01,@obj.computeEdgeForce);
             obj.fixBoundaryElements;
             obj.triangulateNodes;
             obj.assignTriangulationProperties; 
         end
        function smoothGraph(obj,n_cycles)
            for i=1:n_cycles
                newRs=(obj.adj*obj.coords)./obj.degree;
                obj.coords(obj.n_bdry_nodes+1:end,:)=newRs(obj.n_bdry_nodes+1:end,:);
            end
            
              
            obj.fixBoundaryElements;
            obj.triangulateNodes;
            obj.assignTriangulationProperties; 
            
        end

%-------------------------------------------------------------------------
        function showTriangulation(obj,my_fig,color)
            ax1 = axes('Parent', my_fig);
            triplot(obj.elements,obj.coords(:,1),obj.coords(:,2),'Parent',ax1,'Color',color);
        end

        function showMeshQuality(obj)
            obj.assignTriangulationProperties;
             obj.showTriangulation(obj.new_fig,'r');
            edgeVecs  = obj.coords(obj.tri_edges(:,2), :) - obj.coords(obj.tri_edges(:,1), :);
            %twiceArea = abs( AB(:,1).*CA(:,2) - AB(:,2).*CA(:,1) );   % |AB × AC|
            edgeLens  = sqrt(sum(edgeVecs.^2, 2));  % E x 1
            edge_ls= reshape(edgeLens(obj.element_edges), [], 3); %M by 3
            lmax=max(edge_ls,[],2);
            lmin=min(edge_ls,[],2);
            lmid=sum(edge_ls,2)-lmax-lmin;
            min_angle=acos(((lmid-lmax).^2 -lmin.^2)./(2*lmid.*lmax)+1)*180/pi;
            n_elements=size(obj.elements,1);
            % Circumradius R = (a*b*c)/(4*A)
            sum_edges=sum(edge_ls,2)/2; 
            edge_prod=prod(edge_ls,2);
            areas= sqrt( max(0, sum_edges .* (sum_edges - edge_ls(:,1)) .* (sum_edges - edge_ls(:,2)) .* (sum_edges - edge_ls(:,3)) ) );

            circumR = edge_prod ./ (4*areas);
            
            % Inradius r = 2A / (a + b + c)
            inR = areas ./ sum_edges;   % since 2A = twiceArea
        
            angle_ax=axes('Parent', obj.angle_fig);
            histogram(angle_ax,min_angle,20);
             xlabel(angle_ax,'distribution of smallest angle in degrees',...
                 'FontName','Ariel Narrow','FontSize',9);
             ylabel(angle_ax,'Number of elements','FontName','Ariel Narrow','FontSize',9);
             xlim(angle_ax,[0,60]);
            txt= 'Value close to 60 indicate equilateral trinagles  ';
             yloc=n_elements/20;
            text(angle_ax,0,yloc,txt,'FontName','Ariel Narrow','FontSize',9);
        
           % now create histogram of length ratio
           length_ax=axes('Parent', obj.length_fig);
            histogram(length_ax ,1-lmin./lmax,20);
            xlabel(length_ax ,'length ratio (1-l_{min}/l_{max}) of element edges',...
                'FontName','Ariel Narrow','FontSize',9);
            ylabel(length_ax ,'Number of elements','FontName','Ariel Narrow','FontSize',9);
            xlim(length_ax ,[0,1]);
            txt= 'Values close to 0 indicate equilateral triangles ';
            yloc=n_elements/20;
            text(length_ax ,0,yloc,txt,'FontName','Ariel Narrow','FontSize',9);

            % create historam of incircles
            radius_ax=axes('Parent', obj.radius_fig);
            histogram(radius_ax, inR,20);
            xlabel(radius_ax ,'incircle radii of the elements',...
                'FontName','Ariel Narrow','FontSize',9);
            ylabel(radius_ax ,'Number of elements','FontName','Ariel Narrow','FontSize',9);
            xlim(radius_ax ,[0,max(inR)]);

            % create historam of circumr/inR 
            ratio_ax=axes('Parent', obj.ratio_fig);
            histogram(ratio_ax ,circumR./inR,20);
            xlabel(ratio_ax ,'incircle radii of the elements',...
                'FontName','Ariel Narrow','FontSize',9);
            ylabel(ratio_ax ,'Number of elements','FontName','Ariel Narrow','FontSize',9);
         
        end

        
 
        function G=computeVectorizedGrad(obj,edgeGradFunction) % H is 3 by 2
            % computes the "force" at every node accummulated as the  gradient of the
            % cost function 
            % it assumes that the cost frunction can be written as the sum
            % of the cost functions of each triangular element
            % It further assumes that the cost function of any triangle
            % just depends upon the edge lengths of the triangles only
            % this allows us to do away with loops and use accumarray
            % instead. Ths considerably improves speed. 
            % find the coords of the edges of all the triangles 
             edge_vecs  = obj.coords(obj.tri_edges(:,2), :) - obj.coords(obj.tri_edges(:,1), :);
             % find their length
             edgeLens  = sqrt(sum(edge_vecs.^2, 2));  % E x 1
             E = reshape(edgeLens(obj.element_edges), [], 3); %M by 3
            % each of the three edges
             % ffind the gradient of the edge function 
             D=edgeGradFunction(E);
             % thnen covert this into gradient 
             % I have to write the formulae here 
             GE=accumarray(obj.element_edges,D(:)); % this is the assembley set for the edge gradients
             node_grad=GE.*edge_vecs;
             M=size(obj.coords,1);
             all_nodes=[obj.tri_edges(:,1); obj.tri_edges(:,2)];
             x_vec=[-node_grad(:,1); node_grad(:,1)];
             y_vec=[-node_grad(:,2); node_grad(:,2)];
             x_sum = accumarray(all_nodes, x_vec, [M 1]);
             y_sum = accumarray(all_nodes, y_vec, [M 1]);
             G=[x_sum,y_sum];
             
        end 
%_________________________________________________________________________
        % these are the gradients of the cost functions that optimize the
        % shape of the triangles., Input is the 3 edge legnths, Out put is
        % the gradient of the cost function in terms of the edge lengths
        % written as a M by 3 matrix. 
        %This will be used as input by the computeVectorizedGrad 
        function D=computeTriangleGrad(~,E)
            psi=((E(:,1)-E(:,2)).^2+(E(:,2)-E(:,3)).^2+(E(:,3)-E(:,1)).^2);% M by 1 
             % this is the tricky part
             k=0.1;
             s=sum(E,2);%M by 1
             dp=(2./s+2*psi./s.^3); % M by 1 \D is derivaative of dP +  a reultsion term thrown in 
             D=(6./s.^2+(k-(dp))./E);% M by 3
        end


        function D=computeEdgeForce(~,E)
            k=10;
            D=-k./(E.^2+1);
        end
%__________________________________________________________________________
        %Helper functions

        function createFigs(obj)
            obj.UIFig=figure;
            obj.UIFig.Position = [100 100 800 800];
            tg=uitabgroup(obj.UIFig);
            obj.new_fig=uitab(tg,"Title","Final Mesh");
            obj.original_fig=uitab(tg,"Title","Starting Mesh");
            obj.angle_fig=uitab(tg,"Title","Mesh Angle Quality");
            obj.length_fig=uitab(tg,"Title","Mesh Length Quality");
            obj.radius_fig=uitab(tg,"Title","incircle Size distribution");
            obj.ratio_fig=uitab(tg,"Title","circumcircle to incircle rato");
        end


        function setTicks(~,ax,xs,ys)
            %obj.UIAxes.Position=[161,1,478,478];
            Nx=length(xs);
            Ny=length(ys);
            for i=1:Nx, labelx{i}=sprintf('%0.1f',xs(i));end;
            for i=1:Ny, labely{i}=sprintf('%0.1f',ys(i));end;
            ax.XMinorGrid = 'off';
            ax.YMinorGrid = 'off';
            ax.XGrid = 'on';
            ax.YGrid = 'on';
            ax.XLimMode = 'manual';
            ax.XLim = [xs(1) xs(end)];
            ax.YLimMode = 'manual';
            ax.YLim = [ys(1) ys(end)];
            ax.XTickMode = 'manual';
            ax.XTick = (xs);
            ax.XTickLabelMode = 'manual';
            ax.XTickLabel = labelx;
            ax.YTickMode = 'manual';
            ax.YTick = (ys);
            ax.XTickLabelMode = 'manual';
            ax.YTickLabel = labely;
            
        end
%--------------------------------------------------------------------------
        
        function [xy,R0]=createBoundaryPoints(obj,curve_name,indata)
            func_name="create"+curve_name;
            try
                [xy,R0]=obj.(func_name)(indata);
            catch exception
                disp("something worng in  : " +func_name +" perhaps a typo in the input file");
                disp(exception.message);
            end
        end
%--------------------------------------------------------------------------
        function [xy,R0] = createPOLYLINE(~,indata)
            indata=[indata,0]; % pad the last one 
            if mod(numel(indata),3) ~= 0           % length not divisible by 3
                 error("myFunc:BadLength", ...
              "Input may be missing elements  (got %d).", numel(indata)-1);
            end
            polygonPoints=reshape(indata,3,[]); % the last is a dummy 
            % Initialize the output points array
            xy = [];
            d=[];
            % Loop through each segment of the polygonal line
            for i = 1:size(polygonPoints,2)-1
                % Get the start and end points of the current segment
                startPoint = polygonPoints(1:2, i);
                endPoint = polygonPoints(1:2,i+1);
                r0=polygonPoints(3,i);
                % Calculate the distance between the start and end points
                segmentLength = norm(endPoint - startPoint);
                numPoints=ceil(segmentLength/r0)+1;
                % Calculate the direction vector of the segment
                directionVector = (endPoint - startPoint) / segmentLength;
                
                % Create equally spaced points along the current segment
                for j = 0:numPoints-2
                    point = startPoint + directionVector * (segmentLength * j / (numPoints-1));
                    xy = [xy, point];
                end
                d=[d,r0];
            end
            R0=mean(d);
            xy=[xy,polygonPoints(1:2,end)];
        end
%--------------------------------------------------------------------------
        function [xy,R0]=createCIRCLE(~,indata)
            R0=indata(1);
            x=indata(2);
            y=indata(3);
            R=indata(4);
            arc_length=abs(indata(6)-indata(5))*pi/180*R;
            n_pts=ceil(arc_length/R0)+1;
            thetas=pi/180*linspace(indata(5),indata(6),n_pts);
            xy=[x;y]+R*[cos(thetas);sin(thetas)];
        end
%--------------------------------------------------------------------------
        function [xy,R0]=createSPLINE(~,indata)
            n_knots=(length(indata)-1)/2;
            xy=reshape(indata(2:end),2,n_knots); 
            arc_length=norm(diff(xy'));
            t0=linspace(0,arc_length,n_knots);
            R0=indata(1);
            tn=linspace(0,arc_length,indata(1));
            ppspline=spline(t0,xy);
            N_points=length(tn);
            Forcev=zeros(1,N_points);
            eta=0.1;
            R0=0;
            for j=1:50
                xy=ppval(ppspline,tn);
                Forcev(:)=0;
                Ls=vecnorm(diff(xy,1,2),1);
                R0=max(0,max(Ls));
                Forcev(2:N_points-1)=diff(Ls);
                Forcev(1)=0;
                Forcev(N_points)=0;
                tn=tn+eta*Forcev;% adjust the t values to take care of the forces
            end  
        end
%--------------------------------------------------------------------------
    end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
end
%%%%%______________________________________________________________________

%%% Testing

