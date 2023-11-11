use std::collections::hash_map::Entry;
use std::collections::HashMap;
use std::mem::size_of;
use std::time::Instant;

use rand::{Rng, SeedableRng};
use rand_chacha::ChaChaRng;

// number of operations to perform on the state
const ROUNDS: usize = 1 << 20;
// the largest number of operations which may be performed between two states
const MAX_STEP_DIFF: usize = 8;
// the size of each state
const STATE_SIZE: usize = 1 << 16;

// the value type recorded in the state
type ValueType = u64;
// the diff, representing the difference between two states
type Diff = HashMap<usize, (ValueType, ValueType)>;
// the state itself
type State = Vec<ValueType>;
// a cache for recording the diffs between multiple states
type DiffCache = Vec<Diff>;
// the log of all the states we have seen so far as encoded in diffs
type StateLog = Vec<Diff>;

fn initial_state() -> State {
    vec![0; STATE_SIZE]
}

// apply the given diff to the state
fn apply_diff(state: &mut State, diff: &Diff) {
    for (&i, &(orig, new)) in diff {
        debug_assert!(state[i] == orig);
        state[i] = new;
    }
}

// recover this state by progressively applying diffs
fn recover_state(log: &StateLog, index: usize) -> State {
    let mut state = initial_state();

    // that's just the initial state
    if index == 0 {
        return state;
    }

    // select the top bits as though we were binary searching
    let mut mask = usize::MAX << (usize::BITS - log.len().leading_zeros() - 1);
    let mut bit = 1 << (mask.trailing_zeros());

    while bit != 0 {
        let index = index & mask;

        // if the bit we are currently looking at is set to zero, don't apply the diff!
        // we would be repeating the previous diff
        if index & bit != 0 {
            apply_diff(&mut state, &log[index - 1]);
        }

        mask >>= 1;
        bit = bit.overflowing_shr(1).0;
        mask |= 1usize << 63;
    }

    state
}

// union the src diff into the destination diff
fn union_diff(dest: &mut Diff, src: &Diff) {
    for (&k, &(expected, new)) in src.iter() {
        match dest.entry(k) {
            Entry::Occupied(mut entry) => {
                let diff = entry.get_mut();
                let old = diff.0;
                debug_assert!(diff.1 == expected);

                // elide this diff, removing unnecessary
                if old == new {
                    entry.remove();
                } else {
                    diff.1 = new;
                }
            }
            Entry::Vacant(entry) => {
                // we haven't seen this index before; it pre-exists us, so add it here
                entry.insert((expected, new));
            }
        }
    }
}

// create a diff from the most recent relevant cached diff
fn append_diff(log: &mut StateLog, cache: &mut DiffCache, mut diff: Diff) {
    let evicted_count = (log.len() + 1).trailing_zeros();
    let mut last_evicted = None;
    for _ in 0..evicted_count {
        last_evicted = cache.pop().or(last_evicted); // allow for new insertions
    }
    debug_assert!(evicted_count == 0 || last_evicted.is_some());

    // update the diffs that remain
    for remaining in cache.iter_mut() {
        union_diff(remaining, &diff);
    }

    // prepare fresh diff
    if let Some(mut cached) = last_evicted {
        union_diff(&mut cached, &diff); // diff is probably smaller
        diff = cached.clone();

        // reinsert the updated old diff
        cache.push(cached);

        // insert fresh diffs since we haven't made any change since the one we just replaced yet
        for _ in 1..evicted_count {
            cache.push(Diff::new());
        }
    }
    log.push(diff);
}

fn main() {
    let mut rng = ChaChaRng::seed_from_u64(0); // init rng with seed 0

    let mut log = StateLog::new();
    let mut state = initial_state();
    let mut cache = DiffCache::new();
    cache.push(Diff::new()); // initialize the cache

    let start_time = Instant::now();

    // initialise
    for _ in 0..ROUNDS {
        // not realistic updating strategy, but it serves a point
        let mut diff = Diff::new();

        for _ in 0..rng.gen_range(0..MAX_STEP_DIFF) {
            let idx = rng.gen_range(0..state.len());
            let value = rng.gen();

            diff.insert(idx, (state[idx], value));
        }

        apply_diff(&mut state, &diff);
        append_diff(&mut log, &mut cache, diff);

        if log.len().is_power_of_two() {
            println!("log now has {} entries", log.len());

            #[cfg(debug_assertions)]
            for (&i, &(old, new)) in log.last().unwrap() {
                assert_eq!(old, 0);
                assert_eq!(new, state[i]);
            }
        }
    }

    let diff = Instant::now() - start_time;

    println!(
        "it took {} seconds to do {} rounds of snapshotting",
        diff.as_secs_f64(),
        ROUNDS
    );

    println!("checking that all states can be recovered successfully (takes a long time for large state size!)");

    let mut rng = ChaChaRng::seed_from_u64(0); // reinit with seed 0 to test
    let mut state = initial_state(); // reset the state

    // test
    for i in 0..ROUNDS {
        // not realistic updating strategy, but it serves a point
        let mut diff = Diff::new();

        for _ in 0..rng.gen_range(0..MAX_STEP_DIFF) {
            let idx = rng.gen_range(0..state.len());
            let value = rng.gen();

            diff.insert(idx, (state[idx], value));
        }

        apply_diff(&mut state, &diff);

        // ensure that we can recover the state from the log at this index (1-indexed)
        let recovered_state = recover_state(&log, i + 1);
        assert_eq!(state, recovered_state)
    }

    let total_stored: usize = log
        .iter()
        .map(|diff| diff.len() * (size_of::<usize>() + 2 * size_of::<ValueType>()))
        .sum();
    let theoretical_stored = log.len() * state.len() * size_of::<ValueType>();

    // if you don't need to verify the state recovery
    let best_stored: usize = log
        .iter()
        .map(|diff| diff.len() * (size_of::<usize>() + size_of::<ValueType>()))
        .sum();

    println!("verified that all states were recovered accurately");
    println!("storage cost of snapshots: {}", theoretical_stored);
    println!(
        "storage cost of diffs: {} ({}%)",
        total_stored,
        100. * (total_stored as f64 / theoretical_stored as f64)
    );
    println!(
        "storage cost of diffs without revert: {} ({}%)",
        best_stored,
        100. * (best_stored as f64 / theoretical_stored as f64)
    );
}
