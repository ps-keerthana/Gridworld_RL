// ============= RL CORE - CONFIGURATION =============
let GRID_SIZE = 4;
let N_STATES = GRID_SIZE * GRID_SIZE;
const N_ACTIONS = 4; // 0: Up, 1: Down, 2: Left, 3: Right
let START_STATE = 0;
let GOAL_STATE = N_STATES - 1;

// Algorithms
let Q_TABLE, SARSA_Q_TABLE, NAIVE_Q_TABLE;

// Rewards
const REWARD_GOAL = 1000;
const REWARD_HOLE = -100;    
const REWARD_STEP = -1;      

// Holes
let HOLE_STATES = [5, 7, 11, 12]; 
const DEFAULT_HOLES = [5, 7, 11, 12];

// Hyperparameters
let LEARNING_RATE = 0.1;
let DISCOUNT_FACTOR = 0.95; 
let MIN_EXPLORATION_RATE = 0.01;
let EXPLORATION_DECAY_RATE = 0.001;

// ============= STATE TRACKING =============
let total_episodes = 0;
let exploration_rate = 1.0;
let is_training = false;
let training_start_time = 0;
let current_state = 0;
let current_visualized_policy = 'q'; 

// Stats History (Rewards)
let q_rewards = [];
let sarsa_rewards = [];
let naive_rewards = [];

// Stats History (Steps)
let q_steps_history = [];
let sarsa_steps_history = [];
let naive_steps_history = [];

// Win Counters
let q_wins = 0;
let sarsa_wins = 0;
let naive_wins = 0;

// Convergence Tracking
let q_converged_episode = null;
let sarsa_converged_episode = null;

// Charts
let rewardChart, stepChart;

// Initialize
resetTables();

function resetTables() {
    Q_TABLE = Array.from({ length: N_STATES }, () => Array(N_ACTIONS).fill(0));
    SARSA_Q_TABLE = Array.from({ length: N_STATES }, () => Array(N_ACTIONS).fill(0));
    NAIVE_Q_TABLE = Array.from({ length: N_STATES }, () => Array(N_ACTIONS).fill(0));
}

// ============= ENVIRONMENT =============
function step(state, action) {
    let row = Math.floor(state / GRID_SIZE);
    let col = state % GRID_SIZE;

    switch (action) {
        case 0: row = Math.max(0, row - 1); break; 
        case 1: row = Math.min(GRID_SIZE - 1, row + 1); break; 
        case 2: col = Math.max(0, col - 1); break; 
        case 3: col = Math.min(GRID_SIZE - 1, col + 1); break; 
    }

    const nextState = row * GRID_SIZE + col;
    let reward = REWARD_STEP;
    let done = false;

    if (nextState === GOAL_STATE) {
        reward = REWARD_GOAL;
        done = true;
    } else if (HOLE_STATES.includes(nextState)) {
        reward = REWARD_HOLE;
        done = true;
    }

    return { nextState, reward, done };
}

// ============= AGENT BRAINS =============
function chooseAction(state, epsilon, qTable) {
    if (Math.random() < epsilon) {
        return Math.floor(Math.random() * N_ACTIONS);
    } else {
        const q_values = qTable[state];
        const max_q = Math.max(...q_values);
        const best_actions = q_values
            .map((q, i) => q === max_q ? i : -1)
            .filter(i => i !== -1);
        return best_actions[Math.floor(Math.random() * best_actions.length)];
    }
}

function getMaxSteps(gridSize) {
    return Math.pow(gridSize, 2) * 2 + 100; 
}

// ============= EPISODE RUNNER =============
function runEpisode() {
    const max_steps = getMaxSteps(GRID_SIZE);

    // --- Q-LEARNING ---
    let state = START_STATE;
    let done = false;
    let steps = 0;
    let q_ep_reward = 0;
    while (!done && steps < max_steps) {
        const action = chooseAction(state, exploration_rate, Q_TABLE);
        const res = step(state, action);
        const max_future_q = Math.max(...Q_TABLE[res.nextState]);
        Q_TABLE[state][action] += LEARNING_RATE * (res.reward + DISCOUNT_FACTOR * max_future_q - Q_TABLE[state][action]);
        q_ep_reward += res.reward;
        state = res.nextState;
        done = res.done;
        steps++;
    }
    if(q_ep_reward > 0) q_wins++; 
    q_rewards.push(q_ep_reward);
    q_steps_history.push(steps); 

    // --- SARSA ---
    state = START_STATE;
    done = false;
    steps = 0;
    let sarsa_ep_reward = 0;
    let action = chooseAction(state, exploration_rate, SARSA_Q_TABLE);
    while (!done && steps < max_steps) {
        const res = step(state, action);
        const nextAction = chooseAction(res.nextState, exploration_rate, SARSA_Q_TABLE);
        SARSA_Q_TABLE[state][action] += LEARNING_RATE * (res.reward + DISCOUNT_FACTOR * SARSA_Q_TABLE[res.nextState][nextAction] - SARSA_Q_TABLE[state][action]);
        sarsa_ep_reward += res.reward;
        state = res.nextState;
        action = nextAction;
        done = res.done;
        steps++;
    }
    if(sarsa_ep_reward > 0) sarsa_wins++;
    sarsa_rewards.push(sarsa_ep_reward);
    sarsa_steps_history.push(steps); 

    // --- NAIVE ---
    state = START_STATE;
    done = false;
    steps = 0;
    let naive_ep_reward = 0;
    while (!done && steps < max_steps) {
        const n_action = chooseAction(state, 0, NAIVE_Q_TABLE); 
        const res = step(state, n_action);
        naive_ep_reward += res.reward;
        state = res.nextState;
        done = res.done;
        steps++;
    }
    if(naive_ep_reward > 0) naive_wins++;
    naive_rewards.push(naive_ep_reward);
    naive_steps_history.push(steps); 

    // --- UPDATES ---
    total_episodes++;
    exploration_rate = Math.max(MIN_EXPLORATION_RATE, exploration_rate - EXPLORATION_DECAY_RATE);

    // Keep buffers small
    if (q_rewards.length > 200) {
        q_rewards.shift(); sarsa_rewards.shift(); naive_rewards.shift();
        q_steps_history.shift(); sarsa_steps_history.shift(); naive_steps_history.shift();
    }
}

// ============= TRAINING LOOP =============
function startTraining() {
    const startBtn = document.getElementById('start-btn');
    const runPolicyBtn = document.getElementById('run-policy-btn');
    const trainingModeEl = document.getElementById('training-mode');
    const episodesInput = document.getElementById('episodes-input');
    
    const maxEpisodes = GRID_SIZE > 8 ? 20000 : 10000;
    const isBatchMode = trainingModeEl && trainingModeEl.value === 'batch';

    if (!isBatchMode && total_episodes >= maxEpisodes) {
        alert("Training completed! Reset the agent to train again.");
        return;
    }

    if (is_training) {
        is_training = false;
        startBtn.textContent = '▶ Resume Training';
        runPolicyBtn.disabled = false; 
        return;
    }

    is_training = true;
    training_start_time = Date.now();
    startBtn.textContent = '⏸ Pause Training';
    runPolicyBtn.disabled = true; 

    let episodesPerStep = episodesInput ? parseInt(episodesInput.value) : 200;
    if (GRID_SIZE >= 10 && episodesPerStep < 500) episodesPerStep = 500;

    const targetEpisodes = isBatchMode 
        ? (total_episodes + episodesPerStep) 
        : maxEpisodes;

    const trainingLoop = () => {
        if (!is_training) return;

        const startTime = performance.now();
        
        while (total_episodes < targetEpisodes) {
            runEpisode();
            if (performance.now() - startTime > 20) break;
        }

        updateDisplay(); 

        if (total_episodes >= targetEpisodes) {
            is_training = false;
            runPolicyBtn.disabled = false; 
            
            if (isBatchMode) {
                startBtn.textContent = '▶ Resume Training';
            } else {
                startBtn.textContent = '✓ Completed';
                startBtn.disabled = true; 
            }
        } else {
            requestAnimationFrame(trainingLoop);
        }
    };

    trainingLoop();
}

// ============= DISPLAY UPDATES (FIXED CONVERGENCE) =============
function updateDisplay() {
    const elapsed = Math.floor((Date.now() - training_start_time) / 1000);
    document.getElementById('episode-count').textContent = total_episodes.toLocaleString();
    document.getElementById('epsilon-display').textContent = exploration_rate.toFixed(4);
    document.getElementById('training-time').textContent = elapsed + 's';

    const getAvg = (arr) => arr.length ? (arr.reduce((a, b) => a + b, 0) / arr.length) : 0;
    
    const qVal = getAvg(q_rewards);
    const sarsaVal = getAvg(sarsa_rewards);
    
    document.getElementById('q-avg').textContent = qVal.toFixed(1);
    document.getElementById('sarsa-avg').textContent = sarsaVal.toFixed(1);
    document.getElementById('naive-avg').textContent = getAvg(naive_rewards).toFixed(1);

    // === FIX FOR CONVERGENCE ===
    // We relax the threshold. Instead of requiring near-perfect scores (which Q-Learning
    // rarely gets during training due to exploration risks), we ask for:
    // 1. High Win Rate (>90%)
    // 2. Decent Average Reward (> 80% of Perfect Run)
    const min_steps = GRID_SIZE * 2;
    const perfect_run_score = REWARD_GOAL - min_steps;
    const convergence_threshold = perfect_run_score * 0.8; 

    // Win Rates
    const getWinRate = (arr) => arr.length ? arr.filter(r => r > 0).length / arr.length : 0;
    const qRate = getWinRate(q_rewards);
    const sarsaRate = getWinRate(sarsa_rewards);

    // Check Q-Learning
    if (qRate > 0.9 && qVal > convergence_threshold && q_converged_episode === null) {
        q_converged_episode = total_episodes;
    }
    // Check SARSA
    if (sarsaRate > 0.9 && sarsaVal > convergence_threshold && sarsa_converged_episode === null) {
        sarsa_converged_episode = total_episodes;
    }

    document.getElementById('q-conv').textContent = 
        q_converged_episode ? `Ep ${q_converged_episode}` : '--';
    
    document.getElementById('sarsa-conv').textContent = 
        sarsa_converged_episode ? `Ep ${sarsa_converged_episode}` : '--';

    document.getElementById('success-rate').textContent = 
        `Q: ${(qRate*100).toFixed(1)}% | SARSA: ${(sarsaRate*100).toFixed(1)}% | Naive: ${(getWinRate(naive_rewards)*100).toFixed(1)}%`;

    updatePolicyVisualization();
    updateCharts();
}

function changeGridSize(newSize) {
    GRID_SIZE = newSize;
    N_STATES = GRID_SIZE * GRID_SIZE;
    GOAL_STATE = N_STATES - 1;
    
    HOLE_STATES = []; 
    document.getElementById('hole-count').textContent = 0;

    // Auto-tune
    if (newSize >= 10) {
        LEARNING_RATE = 0.15;
        EXPLORATION_DECAY_RATE = 0.0001;
        DISCOUNT_FACTOR = 0.995;
        MIN_EXPLORATION_RATE = 0.05;
    } else if (newSize >= 7) {
        LEARNING_RATE = 0.1;
        EXPLORATION_DECAY_RATE = 0.0005;
        DISCOUNT_FACTOR = 0.98;
        MIN_EXPLORATION_RATE = 0.02;
    } else {
        LEARNING_RATE = 0.1;
        EXPLORATION_DECAY_RATE = 0.001;
        DISCOUNT_FACTOR = 0.95;
        MIN_EXPLORATION_RATE = 0.01;
    }

    const setUI = (id, val) => {
        const el = document.getElementById(id);
        if(el) el.value = val;
        const disp = document.getElementById(id + '-value');
        if(disp) disp.textContent = val;
    };
    setUI('learning-rate', LEARNING_RATE);
    setUI('discount-factor', DISCOUNT_FACTOR);
    setUI('exploration-decay', EXPLORATION_DECAY_RATE);
    
    const decayNum = document.getElementById('exploration-decay-number');
    if(decayNum) decayNum.value = EXPLORATION_DECAY_RATE;

    document.body.setAttribute('data-grid', newSize);
    
    resetAgent(); 
    createHoleGrid();
    createGrid();
}

function resetAgent() {
    is_training = false;
    resetTables();

    total_episodes = 0;
    exploration_rate = 1.0;
    
    q_rewards = []; sarsa_rewards = []; naive_rewards = [];
    q_steps_history = []; sarsa_steps_history = []; naive_steps_history = [];
    
    q_wins = sarsa_wins = naive_wins = 0;
    q_converged_episode = null;
    sarsa_converged_episode = null;

    const startBtn = document.getElementById('start-btn');
    startBtn.textContent = '▶ Start Training';
    startBtn.disabled = false;
    document.getElementById('run-policy-btn').disabled = true;
    
    document.querySelectorAll('.agent').forEach(el => el.classList.remove('agent'));
    
    if (rewardChart) {
        rewardChart.data.labels = [];
        rewardChart.data.datasets.forEach(d => d.data = []);
        rewardChart.update();
    }
    if (stepChart) {
        stepChart.data.labels = [];
        stepChart.data.datasets.forEach(d => d.data = []);
        stepChart.update();
    }
    
    updateDisplay();
}

// ============= VISUALIZATION =============
function createGrid() {
    const grid = document.getElementById('grid-container');
    grid.innerHTML = '';
    grid.style.setProperty('--dynamic-grid-size', GRID_SIZE);
    grid.style.gridTemplateColumns = `repeat(${GRID_SIZE}, 1fr)`;
    
    for (let i = 0; i < N_STATES; i++) {
        const cell = document.createElement('div');
        cell.className = 'cell';
        cell.id = `cell-${i}`;

        if (i === START_STATE) {
            cell.classList.add('start');
            cell.textContent = 'S';
        } else if (i === GOAL_STATE) {
            cell.classList.add('goal');
            cell.textContent = 'G';
        } else if (HOLE_STATES.includes(i)) {
            cell.classList.add('hole');
            cell.textContent = 'H';
        }
        grid.appendChild(cell);
    }
}

function createHoleGrid() {
    const holeGrid = document.getElementById('hole-grid-container');
    holeGrid.innerHTML = '';
    holeGrid.style.gridTemplateColumns = `repeat(${GRID_SIZE}, 1fr)`;

    for (let i = 0; i < N_STATES; i++) {
        const cell = document.createElement('div');
        cell.className = 'hole-cell';
        cell.id = `hole-cell-${i}`;
        
        if (i === START_STATE) {
            cell.classList.add('start-cell');
            cell.textContent = 'S';
        } else if (i === GOAL_STATE) {
            cell.classList.add('goal-cell');
            cell.textContent = 'G';
        } else {
            if (HOLE_STATES.includes(i)) cell.classList.add('hole-active');
            cell.textContent = i;
            cell.addEventListener('click', () => {
                if(is_training) return;
                if(HOLE_STATES.includes(i)) HOLE_STATES = HOLE_STATES.filter(h => h !== i);
                else HOLE_STATES.push(i);
                HOLE_STATES.sort((a,b)=>a-b);
                createHoleGrid();
                createGrid();
            });
        }
        holeGrid.appendChild(cell);
    }
    document.getElementById('hole-count').textContent = HOLE_STATES.length;
}

function updatePolicyVisualization() {
    document.querySelectorAll('.policy-arrow').forEach(el => el.remove());
    
    let qTableToUse;
    if (current_visualized_policy === 'q') qTableToUse = Q_TABLE;
    else if (current_visualized_policy === 'sarsa') qTableToUse = SARSA_Q_TABLE;
    else qTableToUse = NAIVE_Q_TABLE;

    for (let s = 0; s < N_STATES; s++) {
        if (s === GOAL_STATE || HOLE_STATES.includes(s)) continue;
        
        const cell = document.getElementById(`cell-${s}`);
        if (!cell) continue;

        const action = chooseAction(s, 0, qTableToUse);
        const arrow = document.createElement('span');
        arrow.className = 'policy-arrow';
        arrow.textContent = ['↑', '↓', '←', '→'][action];
        cell.appendChild(arrow);
    }
}

function runSelectedPolicy() {
    if (is_training) return;
    current_state = START_STATE;
    document.querySelectorAll('.agent').forEach(el => el.classList.remove('agent'));
    document.querySelectorAll('.path-arrow').forEach(el => el.remove());

    let qTableToUse;
    let arrowColor;
    if (current_visualized_policy === 'q') {
        qTableToUse = Q_TABLE;
        arrowColor = 'var(--color-q-learning)';
    } else if (current_visualized_policy === 'sarsa') {
        qTableToUse = SARSA_Q_TABLE;
        arrowColor = 'var(--color-sarsa)';
    } else {
        qTableToUse = NAIVE_Q_TABLE;
        arrowColor = 'var(--color-naive)';
    }

    const step_fn = () => {
        if (HOLE_STATES.includes(current_state) || current_state === GOAL_STATE) {
            document.getElementById(`cell-${current_state}`).classList.add('agent');
            return;
        }

        const action = chooseAction(current_state, 0, qTableToUse); 
        const { nextState } = step(current_state, action);

        const currentCell = document.getElementById(`cell-${current_state}`);
        if (currentCell) {
            const arrow = document.createElement('span');
            arrow.className = 'path-arrow';
            arrow.textContent = ['↑', '↓', '←', '→'][action];
            arrow.style.color = arrowColor;
            arrow.style.position = 'absolute';
            arrow.style.fontSize = 'clamp(16px, 4cqi, var(--font-size-3xl))';
            arrow.style.fontWeight = 'var(--font-weight-bold)';
            arrow.style.textShadow = '0 2px 4px rgba(0, 0, 0, 0.5)';
            arrow.style.zIndex = '15';
            arrow.style.animation = 'pulseArrow 0.3s ease-out';
            arrow.style.top = '50%';
            arrow.style.left = '50%';
            arrow.style.transform = 'translate(-50%, -50%)';
            currentCell.appendChild(arrow);
        }

        document.querySelectorAll('.agent').forEach(el => el.classList.remove('agent'));
        current_state = nextState;
        document.getElementById(`cell-${current_state}`).classList.add('agent');

        if (!HOLE_STATES.includes(current_state) && current_state !== GOAL_STATE) {
            setTimeout(step_fn, 400);
        }
    };
    step_fn();
}

function setupCharts() {
    const ctxR = document.getElementById('rewardChart');
    const ctxS = document.getElementById('winRateChart');
    if (!ctxR || !ctxS) return;

    const common = { 
        responsive: true, 
        maintainAspectRatio: false, 
        animation: false, 
        interaction: { mode: 'index', intersect: false },
        elements: { point: { radius: 0 } },
        scales: {
            x: { grid: { color: 'rgba(148, 163, 184, 0.1)' }, ticks: { color: '#94a3b8' } },
            y: { grid: { color: 'rgba(148, 163, 184, 0.1)' }, ticks: { color: '#94a3b8' } }
        },
        plugins: {
            legend: { labels: { color: '#94a3b8', font: { family: 'Inter' } } }
        }
    };

    rewardChart = new Chart(ctxR, {
        type: 'line',
        data: { labels: [], datasets: [
            { label: 'Q-Learning', borderColor: '#3b82f6', borderWidth: 2, data: [] },
            { label: 'SARSA', borderColor: '#f97316', borderWidth: 2, data: [] },
            { label: 'Naive', borderColor: '#ef4444', borderWidth: 2, data: [] }
        ]},
        options: { 
            ...common, 
            plugins: { ...common.plugins, title: { display: true, text: 'Avg Reward (Higher is Better)', color: '#94a3b8' } } 
        }
    });

    // Steps Chart
    stepChart = new Chart(ctxS, {
        type: 'line',
        data: { labels: [], datasets: [
            { label: 'Q-Learning', borderColor: '#3b82f6', borderWidth: 2, data: [] },
            { label: 'SARSA', borderColor: '#f97316', borderWidth: 2, data: [] },
            { label: 'Naive', borderColor: '#ef4444', borderWidth: 2, data: [] }
        ]},
        options: { 
            ...common, 
            plugins: { ...common.plugins, title: { display: true, text: 'Steps per Episode (Lower is Better)', color: '#94a3b8' } },
            scales: {
                ...common.scales,
                y: { beginAtZero: true, grid: { color: 'rgba(148, 163, 184, 0.1)' }, ticks: { color: '#94a3b8' } } 
            }
        }
    });
}

function updateCharts() {
    if (total_episodes === 0) return;

    rewardChart.data.labels.push(total_episodes);
    if (rewardChart.data.labels.length > 100) rewardChart.data.labels.shift();

    stepChart.data.labels = rewardChart.data.labels;

    const avg = (arr) => arr.length ? arr.reduce((a,b)=>a+b,0)/arr.length : 0;
    
    // Reward Chart
    const dsR = rewardChart.data.datasets;
    dsR[0].data.push(avg(q_rewards));
    dsR[1].data.push(avg(sarsa_rewards));
    dsR[2].data.push(avg(naive_rewards));
    if (dsR[0].data.length > 100) dsR.forEach(d => d.data.shift());
    rewardChart.update('none'); 

    // Steps Chart
    const dsS = stepChart.data.datasets;
    dsS[0].data.push(avg(q_steps_history));
    dsS[1].data.push(avg(sarsa_steps_history));
    dsS[2].data.push(avg(naive_steps_history));
    if (dsS[0].data.length > 100) dsS.forEach(d => d.data.shift());
    stepChart.update('none');
}

function setupHyperparameterSliders() {
    const connect = (id, varName, displayId) => {
        const el = document.getElementById(id);
        if (!el) return;
        if (varName === 'LEARNING_RATE') el.value = LEARNING_RATE;
        if (varName === 'DISCOUNT_FACTOR') el.value = DISCOUNT_FACTOR;
        if (varName === 'EXPLORATION_DECAY_RATE') {
            el.value = EXPLORATION_DECAY_RATE;
            const numInput = document.getElementById('exploration-decay-number');
            if(numInput) numInput.value = EXPLORATION_DECAY_RATE;
        }
        el.addEventListener('input', (e) => {
            const val = parseFloat(e.target.value);
            if (varName === 'LEARNING_RATE') LEARNING_RATE = val;
            if (varName === 'DISCOUNT_FACTOR') DISCOUNT_FACTOR = val;
            if (varName === 'EXPLORATION_DECAY_RATE') {
                EXPLORATION_DECAY_RATE = val;
                document.getElementById('exploration-decay-number').value = val;
            }
            if (varName === 'MIN_EXPLORATION_RATE') MIN_EXPLORATION_RATE = val;
            if(displayId) document.getElementById(displayId).textContent = val;
        });
    };

    connect('learning-rate', 'LEARNING_RATE', 'learning-rate-value');
    connect('discount-factor', 'DISCOUNT_FACTOR', 'discount-factor-value');
    connect('exploration-decay', 'EXPLORATION_DECAY_RATE', 'exploration-decay-value');
    connect('min-exploration', 'MIN_EXPLORATION_RATE', 'min-exploration-value');

    const gridSlider = document.getElementById('grid-size');
    const gridLabel = document.getElementById('grid-size-value');
    if(gridSlider) {
        gridSlider.addEventListener('input', (e) => {
            if(gridLabel) gridLabel.textContent = `${e.target.value}x${e.target.value}`;
        });
        gridSlider.addEventListener('change', (e) => {
            changeGridSize(parseInt(e.target.value));
        });
    }
    
    document.getElementById('clear-holes-btn').addEventListener('click', () => {
        HOLE_STATES = [];
        createHoleGrid();
        createGrid();
    });
    document.getElementById('reset-holes-btn').addEventListener('click', () => {
        HOLE_STATES = [...DEFAULT_HOLES];
        createHoleGrid();
        createGrid();
    });
    
    document.getElementById('run-policy-btn').addEventListener('click', runSelectedPolicy);

    document.querySelectorAll('.btn-visualize').forEach(btn => {
        btn.addEventListener('click', (e) => {
            document.querySelectorAll('.btn-visualize').forEach(b => b.classList.remove('active'));
            e.target.classList.add('active');
            current_visualized_policy = e.target.dataset.policy;
            const names = {'q': 'Q-Learning', 'sarsa': 'SARSA', 'naive': 'Naive'};
            document.getElementById('current-policy').textContent = names[current_visualized_policy];
            updatePolicyVisualization();
        });
    });
}

document.addEventListener('DOMContentLoaded', () => {
    setupHyperparameterSliders();
    document.getElementById('start-btn').addEventListener('click', startTraining);
    document.getElementById('reset-btn').addEventListener('click', resetAgent);
    
    createHoleGrid(); 
    createGrid();
    setupCharts();
});