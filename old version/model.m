
%% Initialization
clear ; close all; clc

% Load training data (created by 'parser.py')
path_theData = 'dataio/theData.dat';

if(exist(path_theData))
   [data_class, data_tf, data_words] = textread(path_theData, '%d %d %s');
else
	% Error: DNE
   disp(['Could not open ' path_theData '. File does not exist.']);
end

path_features = 'dataio/trainingFeatures.dat';
if (exist(path_features))
	data = load(path_features);
else
	% Error: DNE
   disp(['Could not open ' path_theData '. File does not exist.']);

% pattern = '[^a-zA-Z\d\s:]';
% X = zeros(12500, 5001);


% trainPos = dir('train/pos');
% trainPosSize = size(trainPos)(1,1);

% fprintf('\nComputing Features...\n\n');

% for i = 3:trainPosSize
% 	% Progress Check
% 	if (mod(i, 100) == 0)
% 		percent = i / trainPosSize;
% 		disp(strcat(num2str(percent), ' % Done' ));
% 		fflush(stdout);
% 	end

% 	review = fileread(strcat('train/pos/', trainPos(i).name));
% 	review = lower(review);
% 	review = regexprep(review, pattern, '');
% 	tokens = strsplit(review);
% 	tokens_size = size(tokens)(1,2);
	

% 	for j = 1:tokens_size
% 		X(i, end) = 1;
% 		idx = find ( strcmp(data_words, tokens{j}) );
% 		if !isempty(idx)
% 			X(i, idx) += 1;
% 		end
% 	end

% end
