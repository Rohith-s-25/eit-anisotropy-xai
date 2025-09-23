clear;
clc;
numSamples = 1000;
saveDir = 'DN_Matrix_Batch_new';
if ~exist(saveDir, 'dir')
    mkdir(saveDir);
else
    disp('Warning: Directory already exists.');
end
domainRadius = 28;
numElectrodes = 16;
rng(123); % For reproducibility

for n = 1:numSamples
    model = createpde();
    gd = [1; 0; 0; domainRadius];
    ns = char('C1'); 
    ns = ns';
    sf = 'C1';
    dl = decsg(gd, sf, ns);
    geometryFromEdges(model, dl);
    generateMesh(model, 'Hmax', 1);

    % Random inclusion params with closer values for realism
    xInc = (rand * 2 - 1) * 10;
    yInc = (rand * 2 - 1) * 10;
    rInc = 3.5 + rand * 4.5; % Smaller, range overlaps
    isAniso = rand > 0.5;
    label = double(isAniso);
    lambda_tank = 1.45 + 0.3 * randn; % Slight randomness
    lambda_inc = lambda_tank + 2.5 + rand * 2; % Reduced contrast

    % Anisotropy orientation
    theta_aniso = rand * pi; % Random principal axis orientation

    specifyCoefficients(model, 'm', 0, 'd', 0, ...
        'c', @(location, state) conductivity_realistic(location, lambda_tank, lambda_inc, ...
        xInc, yInc, rInc, isAniso, theta_aniso), ...
        'a', 0, 'f', 0);

    % Random electrode placement jitter
    theta = linspace(0, 2*pi, numElectrodes+1); 
    theta(end) = [];
    theta = theta + 0.05*randn(1, numElectrodes); % Each electrode randomly jittered

    DN = zeros(numElectrodes);
    for i = 1:numElectrodes
        V = trigonometricPattern(i, theta);
        applyBoundaryCondition(model, 'dirichlet', 'Edge', 1:model.Geometry.NumEdges, ...
            'u', @(location, state) voltageBC(location, V, theta));
        result = solvepde(model);
        [gradx, grady] = evaluateGradient(result);
        normalFlux = gradx .* cos(theta(i)) + grady .* sin(theta(i));
        for j = 1:numElectrodes
            DN(i, j) = sum(normalFlux);
        end
    end

    % Add higher, structured noise
    noiseLevel = 0.05 + 0.05*rand; % Random noise strength
    DN_noisy = DN + noiseLevel * randn(size(DN));

    % Simulate missing electrode or measurement dropouts
    if rand < 0.15
        missing_idx = randi([1,numElectrodes],1); % one random electrode
        DN_noisy(missing_idx,:) = NaN;
        DN_noisy(:,missing_idx) = NaN;
    end

    % Simulate boundary deformation
    if rand < 0.10
        DN_noisy = DN_noisy + 0.03*randn(size(DN_noisy)); % more noise for boundary issues
    end

    filename = sprintf('%s/sample_%03d.mat', saveDir, n);
    save(filename, 'DN_noisy', 'label');
end
disp('All samples generated.');

%% --- Helper Functions ---

function c = conductivity_realistic(location, lambda_tank, lambda_inc, x0, y0, r, isAniso, theta_aniso)
    % Always return a 4-row matrix, even for isotropic
    x = location.x; 
    y = location.y;
    dist = sqrt((x - x0).^2 + (y - y0).^2);
    inside = dist <= r;
    Npts = length(x);
    c = zeros(4, Npts);

    if isAniso
        a = 1 + 0.5 * randn;
        b = 1 + 0.5 * randn;
        rot = [cos(theta_aniso), -sin(theta_aniso); sin(theta_aniso), cos(theta_aniso)];
        aniso_matrix = rot * diag([lambda_inc*a, lambda_inc*b]) * rot';
        % Set background to lambda_tank
        c(:, :) = lambda_tank; 
        if any(inside)
            c(1, inside) = aniso_matrix(1,1);
            c(2, inside) = aniso_matrix(1,2);
            c(3, inside) = aniso_matrix(2,1);
            c(4, inside) = aniso_matrix(2,2);
        end
    else
        % For isotropic: diagonal terms set to lambda_inc inside inclusion, rest zero
        c(:, :) = lambda_tank; % background everywhere
        c(1, inside) = lambda_inc;
        c(4, inside) = lambda_inc;
        c(2, :) = 0; % off-diagonal zero everywhere
        c(3, :) = 0;
    end
end

function V = trigonometricPattern(k, theta)
    if mod(k, 2) == 1
        V = cos(((k+1)/2) * theta);
    else
        V = sin((k/2) * theta);
    end
end

function v = voltageBC(location, Vpattern, theta_electrodes)
    x = location.x; 
    y = location.y;
    theta_pts = atan2(y, x);
    theta_pts(theta_pts < 0) = theta_pts(theta_pts < 0) + 2*pi;
    v = zeros(size(x));
    for i = 1:length(theta_pts)
        [~, idx] = min(abs(theta_pts(i) - theta_electrodes));
        v(i) = Vpattern(idx);
    end
end
