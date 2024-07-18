from typing import Any, Optional, Callable

import qiskit.qasm2
from qukit import Dispatcher
from qukit.circuits import VirtualCircuit
from qiskit import QuantumCircuit
from pennylane import qml
import json, uuid, traceback, datetime, os, hashlib
from os.path import join, dirname
from dotenv import load_dotenv
from time import process_time 
import numpy as np


# dotenv_path = join(dirname(__file__), '.env')
# load_dotenv(dotenv_path)

TIME_CUTTING = "time_cutting"
TIME_DISPATCH = "time_dispatch"
TIME_EXECUTION = "time_execution"
TIME_SIMULATION = "time_simulation"
TIME_MERGE = "time_merge"
TIME_EXPECTED_VALUES = "time_expected_values"
TIME_SEW = "time_sew"
TIME_TOTAL = "time_total"
TIME_EXECUTION_RETRIES = "time_execution_retries"

def hash_circuit(qasm):
    return str(hashlib.md5(qasm.encode()).hexdigest())

def pennylane_to_qasm(circuit: qml.QNode) -> str:
    if hasattr(circuit, "tape"):
        c = circuit.tape.to_openqasm()
        
    else:
        c= circuit.to_openqasm()
    
    #print()
    #print("CIRCUIT: "+c)
    #print("HASH: "+str(hash_circuit(c)))
    #print()
    return c

def pennylane_to_vc(circuit: qml.QNode) -> VirtualCircuit:
    qasm = pennylane_to_qasm(circuit)
    return VirtualCircuit(qasm, metadata={"circuit_name":hash_circuit(qasm)})

def qasm_to_pennylane(qasm: str) -> Callable:
    qasm_circuit = qml.from_qasm(qasm)
    def fun():
        qasm_circuit()
    return fun

def push_obs(qasm_circuit, observable_string):
    penny_circuit = qasm_to_pennylane(qasm_circuit)
    def fun():
        penny_circuit()
        return qml.expval(qml.pauli.string_to_pauli_word(observable_string))
    qs = qml.tape.make_qscript(fun)()
    qubits = len(QuantumCircuit.from_qasm_str(qasm_circuit).qubits)
    circuit_name = str(hash_circuit(qasm_circuit))

    vc = VirtualCircuit(pennylane_to_qasm(qs), metadata={"circuit_name":circuit_name})

    return [(vc, qubits)], circuit_name

def pennylane_cut(circuit, observables, cut_params):
    cut_info = {}
    obs =  [qml.expval(qml.pauli.string_to_pauli_word(observables))]
    penny_circ = qasm_to_pennylane(circuit)
    qs = qml.tape.make_qscript(penny_circ)()
    ops= qs.operations
    
    uncut_tape = qml.tape.QuantumTape(ops, obs)

    #Se ci sono i num_fragments, si usano, altrimenti si usa la strategia
    graph = qml.qcut.tape_to_graph(uncut_tape)
    if 'num_fragments' in cut_params:
        cut_graph = qml.qcut.find_and_place_cuts(
            graph = graph,
            num_fragments=cut_params['num_fragments'],
        )
    else:
        cut_graph = qml.qcut.find_and_place_cuts(
            graph = qml.qcut.tape_to_graph(uncut_tape),
            cut_strategy = qml.qcut.CutStrategy(**cut_params),
        )

    qml.qcut.replace_wire_cut_nodes(cut_graph)
    fragments, communication_graph = qml.qcut.fragment_graph(cut_graph)
    fragment_tapes = [qml.qcut.graph_to_tape(f) for f in fragments]

    cut_info["num_fragments"] = len(fragment_tapes)

    cut_info["fragments_qubits"] = [len(tape.wires) for tape in fragment_tapes]
    # Creation of fragments varations
    expanded = [qml.qcut.expand_fragment_tape(t) for t in fragment_tapes]
    
    configurations = []
    prepare_nodes = []
    measure_nodes = []
    for tapes, p, m in expanded:
        configurations.append(tapes)
        prepare_nodes.append(p)
        measure_nodes.append(m)

    tapes_len = [(tape,len(tape.wires)) for c in configurations for tape in c]
    #tapes è lista coppie (tape, numero di qubit del tape)

    num_variations = 0
    variatons = []
    for c in configurations:
        n = 0
        for tape in c:
            num_variations += len(tape.measurements)
            n += len(tape.measurements)
        variatons.append(n)
    cut_info["num_variations"] = num_variations
    cut_info["variations"] = variatons

    return tapes_len, communication_graph, prepare_nodes, measure_nodes, cut_info

def tapes_to_vc(tapes):
    vc_fragments = []
    tapes_info = []
    for tape, tape_qubits in tapes:
        frag_list = []
        for expval in tape.measurements:
            _tape = qml.tape.QuantumTape(ops=tape.operations, measurements=[expval])
            _frag = pennylane_to_vc(_tape)
            vc_fragments.append((_frag, tape_qubits))
            frag_list.append((_tape, _frag.metadata["circuit_name"]))
        tapes_info.append(frag_list)

    return vc_fragments, tapes_info

def calculate_shots(vcs_len, shots_assignment):
    vcs_shots = []
    assignment = shots_assignment[0]
    if assignment == "exponential":
        shots_coefficient = shots_assignment[1]["shots_coefficient"]
        for fragment, qubits in vcs_len:
            shots = shots_coefficient * 2 ** qubits
            vcs_shots.append((fragment, shots))
    elif assignment == "constant":
        shots = shots_assignment[1]["shots"]
        for fragment, qubits in vcs_len:
            vcs_shots.append((fragment, shots))
    elif assignment == "linear":
        shots_coefficient = shots_assignment[1]["shots_coefficient"]
        for fragment, qubits in vcs_len:
            shots = shots_coefficient * qubits
            vcs_shots.append((fragment, shots))
    elif assignment == "fragment_constant":
        shots = shots_assignment[1]["shots"]
        for fragment, _ in vcs_len:
            vcs_shots.append((fragment, shots//len(vcs_len)))
    elif assignment == "exponential_divided":
        qubits = shots_assignment[1]["qubit"]
        shots_coefficient = shots_assignment[1]["shots_coefficient"]
        shots = shots_coefficient * 2 ** qubits
        for fragment, _ in vcs_len:
            vcs_shots.append((fragment, shots//len(vcs_len)))
    else:
        raise Exception("Invalid shots assignment")
    return vcs_shots

def create_single_dispatch(dispatch, fragment,provider, backend, shots):
    if provider not in dispatch:
        dispatch[provider] = {}
    if backend not in dispatch[provider]:
        dispatch[provider][backend] = []
    dispatch[provider][backend].append((fragment, shots))
    return dispatch

def create_dispatch(vcs_fragments, provider_backend_couple, shots_assignment, split_fun):
    vcs_shots = calculate_shots(vcs_fragments, shots_assignment)

    dispatch = {}
    splitted_coefficients = None
    for fragment, shots in vcs_shots:
        if split_fun:
            splitted, splitted_coefficients = split_fun(provider_backend_couple, shots)
            for provider, backend, split_shots in splitted:
                dispatch = create_single_dispatch(dispatch, fragment, provider, backend, split_shots)
        else:
            for provider, backend in provider_backend_couple:
                dispatch = create_single_dispatch(dispatch, fragment, provider, backend, shots)  
    return dispatch, splitted_coefficients


def sample_counts(counts, shots):
    total_shots = sum(counts.values())
    p = [v/total_shots for v in counts.values()]
    c = np.random.multinomial(shots,p)
    
    return {k: int(v) for k,v in zip(counts.keys(), c)}

def simulate_dispatch(dispatch, total_counts):
    results = {}
    for provider in dispatch:
        results[provider] = {}
        for backend in dispatch[provider]:
            results[provider][backend] = []
            for vc,shots in dispatch[provider][backend]:
                for c_id, counts in total_counts[provider][backend]:
                    if c_id != vc.metadata["circuit_name"]:
                        continue
                    # print(f"Simulating {shots} shots of {c_id} from {provider}.{backend} with {counts}")
                    results[provider][backend].append([c_id, sample_counts(counts, shots)])
                    
    return results

def previous_experiments(metadata, subdir):
    try:
        qubit = metadata["run"]["qubit"]
        cicuit_name = metadata["run"]["circuit"]
        observable = metadata["run"]["observable"]
        num_fragments = ""
        penny = ""
        if "no_cc" not in subdir:
            num_fragments = "_"+str(metadata["run"]["cut_tool"][1])
            penny = "_pennylane"
        filename = f'{cicuit_name}_{qubit}_{observable}{penny}{num_fragments}.json'
        print("Filename prev exps: ", filename, flush=True)

        counts = None
        EXP_DIR = metadata["run"]["exp_dir"]
        file_open = EXP_DIR+subdir+filename
        print("File open prev exps: ", file_open, flush=True)
        with open(file_open, "r") as f:
            data = json.load(f)
            counts = data["intermediate_results"]["counts"]
        return counts
    except Exception as e:
        print("No Prev Exps, Exception: "+repr(e), flush=True)
        print(traceback.format_exc(), flush=True)
        return None

def results_to_counts(results_dispatcher):
    try:
        counts_dispatcher = {}
        for provider in results_dispatcher:
            counts_dispatcher[provider] = {}
            for backend in results_dispatcher[provider]:
                counts_dispatcher[provider][backend] = []
                for job in results_dispatcher[provider][backend]:
                    result = job.results[0] 
                    circuit_id = result.circuit.metadata["circuit_name"]
                    _counts = json.loads(json.dumps(result.counts[list(result.counts.keys())[0]]))
                    counts = {k[::-1]:v for k,v in _counts.items()} #TODO: verify endianness for each provider
                    counts_dispatcher[provider][backend].append((circuit_id,counts))  
        return counts_dispatcher
    except AttributeError as e:
        if "'NoneType' object has no attribute 'results'" in repr(e):
            print("IBM ha fallito, rilancio l'esecuzione.")
            return None
        raise e

def prob_calc(counts):
    probs = {}
    for provider in counts:
        if provider not in probs:
            probs[provider] = {}
        for backend in counts[provider]:
            if backend not in probs[provider]:
                probs[provider][backend] = {}
            for circuit_id, _counts in counts[provider][backend]:
                if circuit_id not in probs[provider][backend]:
                    probs[provider][backend][circuit_id] = {}
                probs[provider][backend][circuit_id] = {k: v/sum(_counts.values()) for k,v in _counts.items()}
    return probs

def compute_expected_value(probabilities, observable):
    expected_value = 0
    eigvals = qml.eigvals(observable)
    for state, probability in probabilities.items():
        state_index = int(state, 2)
        expected_value += probability * eigvals[state_index]
    
    return expected_value.real

def expected_values(probs, tape_info):
    _results = []
    for tape in tape_info:
        _sub_results = []
        for _frag, circuit_id in tape:
            # print("----Tape Id "+circuit_id, probs.keys(), flush=True) 
            _sub_results.append(compute_expected_value(probs[circuit_id], _frag.measurements[0].obs))
        _results.append(_sub_results)
    results = []
    for r in _results:
        if len(r) > 1:
            results.append(tuple(r))
        else:
            results.append(r[0])
                
    return tuple(results)

def record_time(times, name, start):
    end = process_time()
    times[name] = (end - start) #in seconds
    return times

#################
def split(couple, shots):
    res = []
    split_coefficients = []
    for provider, backend in couple:
        res.append((provider, backend, shots//len(couple)))
    return res, split_coefficients

def merge(counts):
    probs = {}
    merge_coefficients = []
    for provider in counts:
        for backend in counts[provider]:
            for circuit_id, _counts in counts[provider][backend]:
                if circuit_id not in probs:
                    probs[circuit_id] = {}
                for state, count in _counts.items():
                    if state not in probs[circuit_id]:
                        probs[circuit_id][state] = 0
                    probs[circuit_id][state] += count
    _probs = {}
    for circuit_id in probs:
        _probs[circuit_id] = {k: v/sum(probs[circuit_id].values()) for k,v in probs[circuit_id].items()}

    return _probs, merge_coefficients
#################

def run_exp(
    circuit,
    cutting = False,
    shot_wise = False,
    shots_assignment = None,
    observable_string = None,
    cut_tool = None,
    provider_backend_couple = [],
    split_func = None,
    merg_func = None,
    metadata = None,
):

    #TODO non ignorare cut_tool[0]
    times = {}
    results_dict = {}
    
    print()
    print("----Starting Run ----"+datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"), flush=True)
    print("----Shotwise: ", shot_wise, flush=True)
    print("----Cutting: ", cutting, flush=True)
    if metadata:
        print("----Circuit Name: ", metadata["run"]["circuit"], flush=True)
        print("----Qubit: ", metadata["run"]["qubit"], flush=True)
    print("----Cut Tool: ", cut_tool, flush=True)
    print("----Shots Assignment: ", shots_assignment, flush=True)
    print("----Len Provider Backend Couple: ", len(provider_backend_couple), flush=True)
    if metadata and metadata["run"]:
        run = metadata["run"]
        print("----Split policy: ", run["split_policy"] if "split_policy" in run else "None", flush=True)
        print("----Merge policy: ", run["merge_policy"] if "merge_policy" in run else "None", flush=True)
        #TODO assegnamento condizionale

    if shots_assignment[0] == "exponential_divided":
        dict = shots_assignment[1]
        shots_assignment = ("exponential_divided", {"qubit": metadata["run"]["qubit"], "shots_coefficient": dict["shots_coefficient"]})
    initial_time = process_time()
    #Cut
    if cutting:
        print("----Cutting  start----"+datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"), flush=True)
        start = process_time()
        tapes_len, communication_graph, prepare_nodes, measure_nodes, cut_info = pennylane_cut(circuit, observable_string, cut_tool[1])    
        #Translate tapes to VirtualCircuits
        #vcs_len = pairs (vc, qubits)
        #tapes_info = {tape_hash: [(tape, vc_name)]}
        vcs_len, tapes_info = tapes_to_vc(tapes_len)
        times = record_time(times, TIME_CUTTING, start)
        print("----Fragments: ", cut_info["num_fragments"], flush=True)
        print("----Fragments qubits: ", cut_info["fragments_qubits"], flush=True)
        print("----Variants: ", cut_info["num_variations"], flush=True) #TODO varianti con due osservabili diversi sono contate come una sola
        print("----Variations: ", cut_info["variations"], flush=True)
    else:
        vcs_len, circuit_name= push_obs(circuit, observable_string)
        # print("----Pushed Obs Circuit Name: ", circuit_name, flush=True)

    #Create dispatch
    print("----Dispatch start----"+datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"), flush=True)
    start = process_time()
    dispatch, split_coefficients = create_dispatch(vcs_len, provider_backend_couple, shots_assignment, split_func)
    times = record_time(times, TIME_DISPATCH, start)

    #Check if there are previous experiments
    total_counts = None
    if cutting and shot_wise:
        total_counts = previous_experiments(metadata, "/no_sw/")
    elif not cutting and shot_wise:
        total_counts = previous_experiments(metadata, "/no_cc_no_sw/")
    
    #Simulate or Execute dispatch
    if total_counts:
        print("----Simulate Dispatch start----"+datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"), flush=True)
        start = process_time()
        counts = simulate_dispatch(dispatch, total_counts)
        times = record_time(times, TIME_SIMULATION, start)
    else:
        time_execution_retries = 0.0
        retry = True
        while(retry):
            dispatcher = Dispatcher()
            #Execute the dispatch
            print("----Execution start----"+datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"), flush=True)
            start = process_time()
            execution_results = dispatcher.run(dispatch)
            times = record_time(times, TIME_EXECUTION, start)

            #Counts calculation
            print("----Counts from results start----"+datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"), flush=True)
            counts = results_to_counts(execution_results)
            if counts:
                retry = False
                times[TIME_EXECUTION_RETRIES] = time_execution_retries
            else:
                time_execution_retries += times[TIME_EXECUTION]

    #Merge counts
    if shot_wise:
        print("----Merge start----"+datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"), flush=True)
        start = process_time()
        probs, merge_coefficients = merg_func(counts) #probs = {circuit_id: {state: probability}}
        times = record_time(times, TIME_MERGE, start)
    else:
        probs = prob_calc(counts) #probs = {provider: {backend: {circuit_id: {state: probability}}}}

    # for provider in counts:
        # for backend in counts[provider]:
            # print(f"----Counts keys: {provider} {backend} {[e[0] for e in counts[provider][backend]]}", flush=True)
    #Compute expected values and recompose cutting results if needed
    if cutting and shot_wise:
        print("----Expected Values start----"+datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"), flush=True)
        start = process_time()
        exp_vals = expected_values(probs, tapes_info)
        times = record_time(times, TIME_EXPECTED_VALUES, start)
        print("----Sewing start----"+datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"), flush=True)
        start = process_time()
        r = qml.qcut.qcut_processing_fn(
            exp_vals,
            communication_graph,
            prepare_nodes,
            measure_nodes,
        )
        times = record_time(times, TIME_SEW, start)
    elif cutting and not shot_wise:
        print("----Expected Values and Sewing start----"+datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"), flush=True)
        r = {}
        te = {}
        ts = {}
        for provider in probs:
            if provider not in r:
                r[provider] = {}
                te[provider] = {}
                ts[provider] = {}
            for backend in probs[provider]:
                if backend not in r[provider]:
                    r[provider][backend] = {}
                    te[provider][backend] = {}
                    ts[provider][backend] = {}
                start = process_time()
                exp_vals = expected_values(probs[provider][backend], tapes_info)
                stop = process_time()
                te[provider][backend] = (stop - start)
                start = process_time()
                r[provider][backend] = qml.qcut.qcut_processing_fn(
                                                    exp_vals,
                                                    communication_graph,
                                                    prepare_nodes,
                                                    measure_nodes,
                                                )
                stop = process_time()
                ts[provider][backend] = (stop - start)
        times[TIME_EXPECTED_VALUES] = te
        times[TIME_SEW] = ts
    elif not cutting and shot_wise:  
        obs =  [qml.expval(qml.pauli.string_to_pauli_word(observable_string))]
        print("----Expected Values start----"+datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"), flush=True)
        print("----Probs keys: ", probs.keys(), flush=True)
        print("----Circuit Name: ", circuit_name, flush=True)
        start = process_time()
        r = compute_expected_value(probs[circuit_name], obs[0].obs)
        times = record_time(times, TIME_EXPECTED_VALUES, start)
    else:
        print("----Expected Values start----"+datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"), flush=True)
        r = {}
        te = {}
        for provider in probs:
            if provider not in r:
                r[provider] = {}
                te[provider] = {}
            for backend in probs[provider]:
                if backend not in r[provider]:
                    r[provider][backend] = {}
                    te[provider][backend] = {}
                for circuit_id in probs[provider][backend]:
                    if circuit_id not in r[provider][backend]:
                        r[provider][backend][circuit_id] = {}
                    obs =  [qml.expval(qml.pauli.string_to_pauli_word(observable_string))]
                    start = process_time()
                    r[provider][backend][circuit_id] = compute_expected_value(probs[provider][backend][circuit_name], obs[0].obs)
                    stop = process_time()
                    te[provider][backend][circuit_id] = (stop - start)
        times[TIME_EXPECTED_VALUES] = te
        
    times = record_time(times, TIME_TOTAL, initial_time)
    
    #Building results
    results_dict["parameters"] = {"circuit":circuit, "cutting":cutting, "shot_wise":shot_wise, "observable_string":observable_string, "cut_tool":cut_tool, "shots_assignment":shots_assignment, "provider_backend_couple":provider_backend_couple}
    results_dict["times"] = times
    results_dict["results"] = r
    new_vcs_len = [] #make it json serializable
    for f,q in vcs_len:
        new_vcs_len.append((f.circuit,q))
    
    new_dispatch = {} #make it json serializable
    for provider in dispatch:
        for backend in dispatch[provider]:
            if provider not in new_dispatch:
                new_dispatch[provider] = {}
            if backend not in new_dispatch[provider]:
                new_dispatch[provider][backend] = []
            for frag, shots in dispatch[provider][backend]:
                new_dispatch[provider][backend].append((frag.circuit, shots))
        
    results_dict["intermediate_results"] = {"vcs_len":new_vcs_len, "dispatch":new_dispatch, "counts":counts, "probs":probs}
    if cutting:
        results_dict["cut_info"] = cut_info
        results_dict["intermediate_results"]["exp_vals"] = exp_vals
    if shot_wise:
        results_dict["coefficients"] = {"split":split_coefficients, "merge":merge_coefficients}
    results_dict["metadata"] = metadata

    print("----Ending Run ----"+datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"), flush=True)
    return results_dict

def run(
    circuit,
    cutting = False,
    shot_wise = False,
    shots_assignment = None,
    observable_string = None,
    cut_tool = None,
    provider_backend_couple = [],
    split_func = None,
    merg_func = None,
    metadata = None
):
    cnt = 0
    while True:
        try:
            d = run_exp(
                circuit,
                cutting = cutting,
                shot_wise = shot_wise,
                shots_assignment = shots_assignment,
                observable_string = observable_string,
                cut_tool = cut_tool,
                provider_backend_couple = provider_backend_couple,
                split_func = split_func,
                merg_func = merg_func,
                metadata = metadata
            )
            if "intermediate_results" not in d:
                raise Exception("No intermediate results")
            return d
        except Exception as e:
            cnt += 1
            print(f"!Exception, Retrying {cnt}! "+repr(e), flush=True)
            print("--------------------", flush=True)
            print("--------------------", flush=True)
            print("--------------------", flush=True)
            print(json.dumps({"Exception": repr(e),"traceback":traceback.format_exc(), "metadata": metadata, "parameters": {"circuit":circuit, "cutting":cutting, "shot_wise":shot_wise, "shots_coefficient":shots_coefficient, "observable_string":observable_string, "cut_tool":cut_tool, "provider_backend_couple":provider_backend_couple}, "failed": True, "retry": cnt, "time": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}), flush=True)
            print("--------------------", flush=True)
            print("--------------------", flush=True)
            print("--------------------", flush=True)

if __name__ == "__main__":
    from mqt.bench import get_benchmark
    import qiskit
    
    cicuit_name = "random"
    qubits = 8
    cutting = True
    shot_wise = True
    max_free_wires = 2
    num_fragments = 2

    #cut_args = {"num_fragments": num_fragments}
    cut_args = {"max_free_wires": max_free_wires}

    qiskit_circuit = get_benchmark(cicuit_name, "alg", qubits)
    
    #circuit = qiskit.qasm2.dumps(qiskit_circuit)
    circuit = """OPENQASM 2.0;include "qelib1.inc";qreg q[3];creg c[3];rx(0.531) q[0];ry(0.9) q[1];rx(0.3) q[2];cz q[0],q[1];ry(-0.4) q[0];cz q[1],q[2];measure q[0] -> c[0];measure q[1] -> c[1];measure q[2] -> c[2];"""
    
    #observable = "Z"*qubits
    observable = "XZY"

    #provider_backend_couple=[["ibm_aer", "aer.ibm_brisbane"], ["ibm_aer", "aer.ibm_kyoto"], ["ibm_aer", "aer.ibm_osaka"], ["ibm_aer", "aer.ibm_sherbrooke"]]
    provider_backend_couple=[("ionq","simulator"),("ibm_aer","aer_simulator")]
    
    final = run_exp(circuit,
        cutting=cutting,
        shot_wise=shot_wise,
        observable_string=observable,
        shots_assignment=("constant",{"shots":1000}),
        cut_tool=("pennylane",cut_args),
        provider_backend_couple=provider_backend_couple,
        split_func= split,
        merg_func= merge)

    json.dump(final, open("result.json", "w"), indent=4)
    #Result: Roba che non serve non la metto
    # {'times': {str:float}}
    # {'coefficients': {str:list}} str = split, merge + output funzioni split e merge
    # {'metadata': metadata}
    # {'results': result} = expected value finale
    # {'parameters: parametri di ingresso}
    # {'intermediate_results': vcs_len,dispatch,execution_results,counts,probs,exp_vals
    # {'cut_info': cut_info} se cutting=True, informazioni di cut 