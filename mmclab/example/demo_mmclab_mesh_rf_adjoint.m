%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% MMCLAB - Mesh-based RF (frequency-domain) adjoint Jacobian demo
%
% Same setup as demo_mmclab_mesh_adjoint.m, but with cfg.omega > 0 to
% switch the forward solver into RF mode.
%
% Two code paths are shown side-by-side:
%
%  1. cfg.method = 'grid' (mesh-input voxel-output / DMMC mode):
%     The kernel accumulates a complex phi(r, omega) per slot on a
%     regular voxel grid. The adjoint Jacobian post-processing produces
%     a complex J_mua/J_D per source-detector pair:
%
%         Re(J_mua) = Re(phi_s)*Re(phi_d) - Im(phi_s)*Im(phi_d)
%         Im(J_mua) = Re(phi_s)*Im(phi_d) + Im(phi_s)*Re(phi_d)
%
%     This is the same complex-conjugate-product pattern as
%     mcxlab/example/demo_mcx_adjoint_jacobian.m and rb_femjacobian in
%     redbird-m matlab/rbfemmatrix.cpp.
%
%  2. cfg.method = 'elem' (true mesh output):
%     The mesh-space adjoint kernel runs and returns J_mua/J_D as
%     [nn, Ns*Nd]. However, the BLBadouel mesh ray-tracer in mmc_core.cl
%     does NOT currently accumulate the imaginary part of the fluence
%     (only the rtBLBadouelGrid path does), so the resulting J_mua is
%     real-valued (phase = 0). Use 'grid' mode above when you need the
%     complex/phase information; the mesh-mode amplitude still tracks
%     the correct banana profile.
%
% This file is part of Mesh-based Monte Carlo (MMC) URL:https://mcx.space/mmc
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clear cfg cfg_grid cfg_mesh;

% ---- Common geometry: 60 x 60 x 30 mm box mesh ---------------------
[node, face, elem] = meshabox([0 0 0], [60 60 30], 4);

cfg.nphoton = 3e7;
cfg.node = node;
cfg.elem = elem;
cfg.elemprop = ones(size(cfg.elem, 1), 1);
cfg.srcpos = [30 30 0];
cfg.srcdir = [0  0  1];
cfg.detpos = [20 30 0 1.5
              40 30 0 1.5];
cfg.detdir = [0 0 1 0
              0 0 1 0];
cfg.prop = [0 0 1 1; 0.005 1 0 1.37];
cfg.tstart = 0;
cfg.tend = 5e-9;
cfg.tstep = 5e-9;
cfg.omega = 2 * pi * 100e6;          % 100 MHz RF modulation
cfg.basisorder = 1;
cfg.outputtype  = 'adjoint';
cfg.adjointmode = 0;
cfg.debuglevel  = 'TP';

% ---- Path 1: cfg.method = 'grid' (voxel-output adjoint, full RF complex)
cfg_grid        = cfg;
cfg_grid.method = 'grid';
cfg_grid.steps  = [1 1 1];                % 1 mm voxel grid

fprintf('\n-- Running RF adjoint, grid output (complex J_mua expected) --\n');
f_grid = mmclab(cfg_grid);

fprintf('  grid jmua class: %s   isreal: %d   size: %s\n', ...
        class(f_grid.jmua), isreal(f_grid.jmua), mat2str(size(f_grid.jmua)));
fprintf('  |jmua| range: [%.2e, %.2e]\n', ...
        min(abs(f_grid.jmua(:))), max(abs(f_grid.jmua(:))));

% ---- Path 2: cfg.method = 'elem' (true mesh output, currently real-only)
cfg_mesh        = cfg;
cfg_mesh.method = 'elem';

fprintf('\n-- Running RF adjoint, mesh output (real-only fallback) --\n');
f_mesh = mmclab(cfg_mesh);

fprintf('  mesh jmua class: %s   isreal: %d   size: %s\n', ...
        class(f_mesh.jmua), isreal(f_mesh.jmua), mat2str(size(f_mesh.jmua)));
fprintf('  |jmua| range: [%.2e, %.2e]\n', ...
        min(abs(f_mesh.jmua(:))), max(abs(f_mesh.jmua(:))));

% ---- Plot ----------------------------------------------------------
%  Top row:  grid amplitude + grid phase, S-D pair 1, on the y=30 plane
%  Bottom :  mesh amplitude  (no phase: see header note)              .
figure('Name', 'Mesh RF adjoint Jacobian (mmclab)', ...
       'Position', [60 60 1300 760]);

% Top-left: |J_mua| from grid path (S-D pair 1)
subplot(2, 3, 1);
J_grid = double(f_grid.jmua(:, :, :, 1));     % [Nx, Ny, Nz]
Nx = size(J_grid, 1);
Ny = size(J_grid, 2);
Nz = size(J_grid, 3);
ym = round(Ny / 2);
sl = squeeze(abs(J_grid(:, ym, :)))';
imagesc(0:Nx - 1, 0:Nz - 1, log10(sl + 1e-12));
hold on;
plot(cfg.srcpos(1), 0, 'r^', 'MarkerSize', 9, 'MarkerFaceColor', 'r');
plot(cfg.detpos(1, 1), 0, 'bs', 'MarkerSize', 9, 'MarkerFaceColor', 'b');
xlabel('x (mm)');
ylabel('z (mm)');
title('|J_{\mu_a}|  grid mode  (S-D_1, RF 100MHz)');
colorbar;
set(gca, 'YDir', 'normal');

% Top-middle: arg(J_mua) from grid path (S-D pair 1)
subplot(2, 3, 2);
ph = squeeze(angle(J_grid(:, ym, :)))' * 180 / pi;
imagesc(0:Nx - 1, 0:Nz - 1, ph);
hold on;
plot(cfg.srcpos(1), 0, 'r^', 'MarkerSize', 9, 'MarkerFaceColor', 'r');
plot(cfg.detpos(1, 1), 0, 'bs', 'MarkerSize', 9, 'MarkerFaceColor', 'b');
xlabel('x (mm)');
ylabel('z (mm)');
title('arg(J_{\mu_a})  grid mode  (deg)');
colorbar;
colormap(gca, hsv);
set(gca, 'YDir', 'normal');

% Top-right: |J_mua| pair 2 from grid path
subplot(2, 3, 3);
J_grid2 = double(f_grid.jmua(:, :, :, 2));
sl2 = squeeze(abs(J_grid2(:, ym, :)))';
imagesc(0:Nx - 1, 0:Nz - 1, log10(sl2 + 1e-12));
hold on;
plot(cfg.srcpos(1), 0, 'r^', 'MarkerSize', 9, 'MarkerFaceColor', 'r');
plot(cfg.detpos(2, 1), 0, 'bs', 'MarkerSize', 9, 'MarkerFaceColor', 'b');
xlabel('x (mm)');
ylabel('z (mm)');
title('|J_{\mu_a}|  grid mode  (S-D_2)');
colorbar;
set(gca, 'YDir', 'normal');

% Bottom row: mesh-mode amplitudes (no phase available -- see header note)
for k = 1:size(f_mesh.jmua, 2)
    subplot(2, 3, 3 + k);
    Jamp = abs(double(f_mesh.jmua(:, k)));
    plotmesh([cfg.node, log10(Jamp + 1e-12)], cfg.elem, 'y=30', ...
             'facecolor', 'interp', 'linestyle', 'none');
    view([0 1 0]);
    colorbar;
    title(sprintf('|J_{\\mu_a}|  mesh mode  (S-D_%d, amp only)', k));
end

sgtitle('RF (100 MHz) adjoint Jacobian: grid (complex) vs mesh (real-only)');
