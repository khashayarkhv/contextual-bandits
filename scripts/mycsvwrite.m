%% mycsvwrite.m

% This function generates a csv file with a given header_row and delimiter.

%% Inputs:
%   filename: The filename (or path).
%   data: The array that needs to be saved..
%   header_row: The header_row given as a string, separated by delimiter.
%   delimiter: Delimiter that separates the array elements.
%

function mycsvwrite(filename, data, header_row, delimiter)

fid = fopen(filename, 'w');
fprintf(fid, strcat(header_row,'\n'));
fclose(fid);

dlmwrite(filename, data, '-append', 'delimiter', delimiter);
