function run_optimization_comparison()

clc; clear; close all;

% === Configuration ===
func_names = {'Sphere', 'Schwefel12', 'Rosenbrock'};
func_ids = [1, 2, 6];
D = 11;
runs = 15;
maxIter = 200;

% Function-specific bounds (wider for easier functions)
bounds = {
    [-500, 500];  % Sphere
    [-500, 500];  % Schwefel12
    [-100, 100];  % Rosenbrock
};

all_stats = struct();

for i = 1:length(func_ids)
    func_num = func_ids(i);
    func_name = func_names{i};
    range = bounds{i};
    lb = range(1) * ones(1,D);
    ub = range(2) * ones(1,D);

    objFunc = @(x) benchmark_func(x, func_num);

    fprintf('\n=== Running on Function %d: %s (D=%d) ===\n', func_num, func_name, D);

    % Run all algorithms
    [~, pso_stats] = run_algorithm('PSO', objFunc, D, lb, ub, runs, maxIter);
    [~, ga_stats]  = run_algorithm('GA',  objFunc, D, lb, ub, runs, maxIter);
    [~, sa_stats]  = run_algorithm('SA',  objFunc, D, lb, ub, runs, maxIter);

    % Store stats
    all_stats(i).function = func_name;
    all_stats(i).pso = pso_stats;
    all_stats(i).ga  = ga_stats;
    all_stats(i).sa  = sa_stats;

    % Display summary
    fprintf('\n--- Summary for %s ---\n', func_name);
    display_stats('PSO', pso_stats);
    display_stats('GA',  ga_stats);
    display_stats('SA',  sa_stats);

    % Determine best algorithm by mean
    means = [pso_stats.mean, ga_stats.mean, sa_stats.mean];
    [~, best_idx] = min(means);
    best_algo = {'PSO', 'GA', 'SA'};
    fprintf('✅ Best algorithm for %s: %s (Mean Fitness: %.4f)\n', ...
        func_name, best_algo{best_idx}, means(best_idx));
end
end

%% === Run Algorithm Function ===
function [all_best, stats] = run_algorithm(alg, objFunc, D, lb, ub, runs, maxIter)
    all_best = zeros(runs,1);
    for r = 1:runs
        switch alg
            case 'PSO'
                options = optimoptions('particleswarm','Display','off','MaxIterations',maxIter);
                [~, fval] = particleswarm(objFunc, D, lb, ub, options);
            case 'GA'
                options = optimoptions('ga','Display','off','MaxGenerations',maxIter);
                [~, fval] = ga(objFunc, D, [], [], [], [], lb, ub, [], options);
            case 'SA'
                x0 = lb + rand(1,D).*(ub - lb);
                options = optimoptions('simulannealbnd','Display','off','MaxIterations',maxIter);
                [~, fval] = simulannealbnd(objFunc, x0, lb, ub, options);
        end
        all_best(r) = fval;
    end
    stats.best = min(all_best);
    stats.worst = max(all_best);
    stats.mean = mean(all_best);
    stats.std = std(all_best);
end

%% === Display Stats ===
function display_stats(name, stats)
    fprintf('%s => Best: %.4f | Worst: %.4f | Mean: %.4f | Std: %.4f\n', ...
        name, stats.best, stats.worst, stats.mean, stats.std);
end

%% === Benchmark Function with Shift + Noise ===
function f = benchmark_func(x, func_num)
    if isvector(x)
        x = reshape(x, 1, []);
    end

    shift = 5 * ones(1, size(x,2));  % shift vector

    switch func_num
        case 1  % Sphere (with shift + noise)
            z = x - shift;
            f = sum(z.^2, 2) + 0.01 * randn(size(x,1),1);

        case 2  % Schwefel12 (with shift + noise)
            z = x - shift;
            f = zeros(size(x,1),1);
            for i = 1:size(x,1)
                for j = 1:size(x,2)
                    f(i) = f(i) + sum(z(i,1:j))^2;
                end
            end
            f = f + 0.01 * randn(size(x,1),1);

        case 6  % Rosenbrock (unchanged, no shift)
            f = sum(100*(x(:,2:end) - x(:,1:end-1).^2).^2 + (x(:,1:end-1)-1).^2, 2);

        otherwise
            error('Unsupported function ID');
    end
end