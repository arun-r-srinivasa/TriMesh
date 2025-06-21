     classdef FileIO<handle
    %a class for reading text files for input into FEA programs and related
    %items 
    %  it has a major command called readFilethat takes on object.
    % the object should have a list of field names to read such as "NODES"
    % or "ELEMENTS" etc and a corresponding function setNODES and
    % setELEMENTS
    % it is important that the field names should be identical to the name
    % in the 
    % function 
    % the input file is of the form
    % '*NODES' <----Name of the field 
    % NAME,5, 10, x^2+2*y,.... <------- comma separated list that is read as a
    % character list to be parsed by the object 
    % it can contaition any mix of characters
    % it should end with a *END
   
    % any lise starting with '# s treated as a comment
    % any list starting with a * is treated as a new group of inputs 


    methods ( Static = true )
        
        function readFile(obj)
         [file, location] = uigetfile('*.txt','Select a *.txt file');
        file_name = fullfile(location, file);
        disp(['User selected ', file_name]);
        currentgroup={};
        function_name=" "; 
        fid = fopen(file_name,'r');
            while(~feof(fid))
                 tline = fgetl(fid);
        

                % if it is not a valid line or if it is end of line break 
                % fgetl can return -1 at end of file or an empty string if line is blank
                if ~ischar(tline)
                    break;
                end

                % if it is comment or blank, skip;
                if isempty(tline)
                    continue;  % skip empty lines if needed
                end
        
                % Skip lines that start with '#'
                if tline(1) == '#'
                    continue;
                end
                
                % Check for lines that start with '*'
                if tline(1) == '*'
                    % if group complete and new group starting
                    if (~isempty(currentgroup))
                        try 
                            % call the function with old name and current
                            % group of data
                            obj.(function_name)(currentgroup); 
                        catch exception
                            disp("problem with : " +function_name +" perhaps a typo in the input file");
                            disp(exception.message);
                            break
                        end
                        % reset buffer
                        currentgroup={};% restart 
                    end
                   % create the new function name 
                   if(strcmp(tline,'*END')) % if this is the end
                        break;
                    end
                    function_name="set"+tline(2:end); % 
               
                else
                    % Assume that this is information to be added to group
                    currentgroup{end+1} = tline;
                end
               
            end
             fclose(fid);
        end

         function saveToFile(varargin)
            % saveMyDataToFile: Prompts the user for a file name/location and writes data to that file.
            %
            %   dataToWrite: (optional) data string or array you want to write out.
            %                In this example, we assume a string for simplicity.
        
            % Prompt user for file name and path (for a .txt file as an example)
            [fileName, folderPath] = uiputfile('*.txt', 'Save Data As');
    
             % If the user cancels the dialog, fileName or folderPath will be 0
            if isequal(fileName, 0) || isequal(folderPath, 0)
                disp('User cancelled file selection.');
                return;
            end
    
            % Create the full file path
            fullFilePath = fullfile(folderPath, fileName);

            fid = fopen(fullFilePath, 'w');
    
            if fid == -1
                error('Failed to create file: %s', fullFilePath);
            end
            % Iterate over each input
            for i = 1 : nargin
                
                % Get the variable name if it exists, otherwise create a fallback
                varName = inputname(i);
                if isempty(varName)
                    varName = sprintf('Argument%d', i);
                end
                
                % Get the actual content of the i-th argument
                val = varargin{i};
                
                % --- Write header: variable name
                fprintf(fid, 'Variable Name: %s\n', varName);
                
                % --- Write contents
                if isnumeric(val)
                    % If it's numeric, we can display size, then each row
                    [r, c] = size(val);
                    fprintf(fid, 'Size: %dx%d\n', r, c);
                    fprintf(fid, 'Contents:\n');
                    for row = 1:r
                        fprintf(fid, '%g ', val(row,:)); % Print each row
                        fprintf(fid, '\n');
                    end
                elseif ischar(val) || isstring(val)
                    % Strings or character arrays
                    fprintf(fid, 'Contents: %s\n', val);
                elseif iscell(val)
                    % If it's a cell array, handle as needed
                    fprintf(fid, 'Cell array contents (showing each cell):\n');
                    disp(val); % or parse further as needed
                else
                    % Fallback for other types
                    fprintf(fid, 'Contents (class: %s):\n', class(val));
                    disp(val); % or customize printing
                end
                fprintf(fid, '#\n');  % Blank line for readability
            end
    
            % Close the file
            fclose(fid);
            
            fprintf('Data saved to: %s\n', fullFilePath);
                
        end  
    end
end