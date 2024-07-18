INT_GIT = False 

import configparser
import sys
import json
from tqdm import tqdm
import random
import os

from mqt.bench import get_benchmark

import datetime

import pennylane as qml
import qiskit
import qiskit.qasm2

import pandas as pd

from itertools import chain, combinations

from dotenv import load_dotenv
load_dotenv()

import path
import sys

# directory reach
directory = path.Path(__file__).absolute()

# setting path
sys.path.append(directory.parent.parent)

from run import run

EXPS_DIR = "./exps"
if not os.path.exists(EXPS_DIR):
    os.makedirs(EXPS_DIR)
    
def powerset(iterable):
    "powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(len(s)+1))

def qasm_to_pennylane(qasm: str, observable_string):
    qasm_circuit = qml.from_qasm(qasm)
    def fun():
        qasm_circuit()
        return qml.expval(qml.pauli.string_to_pauli_word(observable_string))
    return fun

def compute_theoretical_exp_value(qasm_circuit, observable_string):
    qc = qiskit.QuantumCircuit.from_qasm_str(qasm_circuit)
    nqubits = qc.num_qubits
    dev = qml.device('default.qubit', wires=nqubits)    
    node = qml.QNode(qasm_to_pennylane(qasm_circuit, observable_string), dev)
    return float(node())


def fair_policy(backends):
    coefficients = {}
    for provider,backend in backends:
        if provider not in coefficients:
            coefficients[provider] = {}
        coefficients[provider][backend] = 1/len(backends)
    return coefficients

def random_policy(backends):
    coefficients = {}
    for provider,backend in backends:
        if provider not in coefficients:
            coefficients[provider] = {}
        coefficients[provider][backend] = random.random()
    total = sum([sum(coefficients[provider].values()) for provider in coefficients])
    for provider in coefficients:
        for backend in coefficients[provider]:
            coefficients[provider][backend] /= total
    return coefficients

import pennylane as qml
from pennylane import numpy as np
import networkx as nx

from typing import List, Optional, Tuple


def clustered_chain_graph(
    n: int, r: int, k: int, q1: float, q2: float, seed: Optional[int] = None
) -> Tuple[nx.Graph, List[List[int]], List[List[int]]]:
    """
    Function to build clustered chain graph

    Args:
        n (int): number of nodes in each cluster
        r (int): number of clusters
        k (int): number of vertex separators between each cluster pair
        q1 (float): probability of an edge connecting any two nodes in a cluster
        q2 (float): probability of an edge connecting a vertex separator to any node in a cluster
        seed (Optional[int]=None): seed for fixing edge generation

    Returns:
        nx.Graph: clustered chain graph
    """

    if r <= 0 or not isinstance(r, int):
        raise ValueError("Number of clusters must be an integer greater than 0")

    clusters = []
    for i in range(r):
        _seed = seed * i if seed is not None else None
        cluster = nx.erdos_renyi_graph(n, q1, seed=_seed)
        nx.set_node_attributes(cluster, f"cluster_{i}", "subgraph")
        clusters.append(cluster)

    separators = []
    for i in range(r - 1):
        separator = nx.empty_graph(k)
        nx.set_node_attributes(separator, f"separator_{i}", "subgraph")
        separators.append(separator)

    G = nx.disjoint_union_all(clusters + separators)

    cluster_nodes = [
        [n[0] for n in G.nodes(data="subgraph") if n[1] == f"cluster_{i}"] for i in range(r)
    ]
    separator_nodes = [
        [n[0] for n in G.nodes(data="subgraph") if n[1] == f"separator_{i}"] for i in range(r - 1)
    ]

    rng = np.random.default_rng(seed)

    for i, separator in enumerate(separator_nodes):
        for s in separator:
            for c in cluster_nodes[i] + cluster_nodes[i + 1]:
                if rng.random() < q2:
                    G.add_edge(s, c)

    return G, cluster_nodes, separator_nodes


def get_qaoa_circuit(
    G: nx.Graph,
    cluster_nodes: List[List[int]],
    separator_nodes: List[List[int]],
    params: Tuple[Tuple[float]],
    layers: int = 1,
) -> qml.tape.QuantumTape:
    """
    Function to build QAOA max-cut circuit tape from graph including `WireCut` 
    operations
    
    Args:
        G (nx.Graph): problem graph to be solved using QAOA
        cluster_nodes (List[List[int]]): nodes of the clusters within the graph
        separator_nodes (List[List[int]]): nodes of the separators in the graph
        params (Tuple[Tuple[float]]): parameters of the QAOA circuit to be optimized
        layers (int): number of layer in the QAOA circuit
        
    Returns:
        QuantumTape: the QAOA tape containing `WireCut` operations
    """
    wires = len(G)
    r = len(cluster_nodes)

    with qml.tape.QuantumTape() as tape:
        for w in range(wires):
            qml.Hadamard(wires=w)

        for l in range(layers):
            gamma, beta = params[l]

            for i, c in enumerate(cluster_nodes):
                if i == 0:
                    current_separator = []
                    next_separator = separator_nodes[0]
                elif i == r - 1:
                    current_separator = separator_nodes[-1]
                    next_separator = []
                else:
                    current_separator = separator_nodes[i - 1]
                    next_separator = separator_nodes[i]

                #for cs in current_separator:
                #    qml.WireCut(wires=cs)

                nodes = c + current_separator + next_separator
                subgraph = G.subgraph(nodes)

                for edge in subgraph.edges:
                    qml.IsingZZ(2*gamma, wires=edge) # multiply param by 2 for consistency with analytic cost

            # mixer layer
            for w in range(wires):
                qml.RX(2*beta, wires=w)


            # reset cuts
            #if l < layers - 1:
            #    for s in separator_nodes:
            #        qml.WireCut(wires=s)

        #[qml.expval(op) for op in cost.ops if not isinstance(op, qml.ops.identity.Identity)]
        observable = "Z"*wires
        [qml.expval(qml.pauli.string_to_pauli_word(observable))]

    return tape

def generate_qaoa_maxcut_circuit(n: int, r: int, k: int, layers: int = 1, q1: float = 0.7, q2: float = 0.3, seed: Optional[int] = None) -> str:
    """
    Function to generate QAOA max-cut circuit tape from graph including `WireCut` 
    operations
    
    Args:
        n (int): number of nodes in each cluster
        r (int): number of clusters
        k (int): number of vertex separators between each cluster pair
        layers (int): number of layer in the QAOA circuit
        q1 (float): probability of an edge connecting any two nodes in a cluster
        q2 (float): probability of an edge connecting a vertex separator to any node in a cluster
        seed (Optional[int]=None): seed for fixing edge generation
        
    Returns:
        QuantumTape: the QAOA tape containing `WireCut` operations
    """
    G, cluster_nodes, separator_nodes = clustered_chain_graph(n, r, k, q1, q2, seed)
    params = ((0.1, 0.2), (0.3, 0.4))
    return get_qaoa_circuit(G, cluster_nodes, separator_nodes, params, layers)


def shots_from_dispatch(dispatch):
    shots = []
    for provider in dispatch:
        for backend in dispatch[provider]:
            shots.append([])
            for _, s in dispatch[provider][backend]:
                # print(provider, backend, s, flush=True)
                shots[-1].append(s)
    return shots
if __name__ == "__main__":
    print("Starting...", flush=True)
    if len(sys.argv) > 1:
        action = sys.argv[1]
    else:
        action = "all"
        
    print("Action: ", action, flush=True)
        
        
    if action in ("all", "exp", "run"):
        config_file = None
        
        if len(sys.argv) > 2:
            config_file = sys.argv[2]
        else:
            config_file = "./config.ini"
        
        config = configparser.ConfigParser()
        config.read(config_file)
        
        print("Config File: ", config_file, flush=True)
            
        now = datetime.datetime.now()
        EXP_DIR = os.path.join(EXPS_DIR, now.strftime("%Y-%m-%d_%H-%M-%S"))
        os.makedirs(EXP_DIR)

        NO_CC_NO_SW_DIR = os.path.join(EXP_DIR, "no_cc_no_sw")
        os.makedirs(NO_CC_NO_SW_DIR)

        NO_CC_DIR = os.path.join(EXP_DIR, "no_cc")
        os.makedirs(NO_CC_DIR)

        NO_SW_DIR = os.path.join(EXP_DIR, "no_sw")
        os.makedirs(NO_SW_DIR)

        CC_SW_DIR = os.path.join(EXP_DIR, "cc_sw")
        os.makedirs(CC_SW_DIR)

        PERF_EXP_VAL_DIR = os.path.join(EXP_DIR, "perf_exp_val")
        os.makedirs(PERF_EXP_VAL_DIR)
        
        with open(os.path.join(EXP_DIR, "config.ini"), "w") as f:
            config.write(f)
            
        with open(os.path.join(EXP_DIR, "exp.py"), "w") as f:
            with open(sys.argv[0], "r") as f2:
                f.write(f2.read())
                
        with open(os.path.join(EXP_DIR, "run.py"), "w") as f:
            with open("run.py", "r") as f2:
                f.write(f2.read())
        
        circuits_configuration_file = json.loads(config["SETTINGS"]["circuits"])
        all_backends = json.loads(config["SETTINGS"]["backends"])
        cut_strategies = json.loads(config["SETTINGS"]["cut_strategies"])
        shotwise_policies = json.loads(config["SETTINGS"]["shotwise_policies"])
        shots_assignment = json.loads(config["SETTINGS"]["shots_assignment"])
        shots_assignment_name = shots_assignment[0]
        shots_assignment_parms = {}
        for i in range(1, len(shots_assignment)):
            shots_assignment_parms[shots_assignment[i][0]] = shots_assignment[i][1]
        shots_assignment = ((shots_assignment_name, shots_assignment_parms))
    
        circuits_configurations = json.load(open(circuits_configuration_file, "r"))
        
        with open(os.path.join(EXP_DIR, "circuits_configurations.json"), "w") as f:
            json.dump(circuits_configurations, f)
        
        for circuit_conf in tqdm(circuits_configurations, desc="Circuits"):
            
            penny_circuit = generate_qaoa_maxcut_circuit(**circuit_conf)
            circuit_qasm = str(penny_circuit.to_openqasm())
            circuit_qubits = penny_circuit.num_wires
            circuit_name = f"n{circuit_conf['n']}_r{circuit_conf['r']}_k{circuit_conf['k']}_p{circuit_conf['layers']}_s{circuit_conf['seed']}"
            
            metadata = {
                "config":{
                    "circuits_configuration_file": circuits_configuration_file,
                    "circuits_configurations": circuits_configurations,
                    "circuits": [],
                    "circuits_names": [],
                    "backends": all_backends,
                    "cut_strategies": cut_strategies,
                    "shotwise_policies": shotwise_policies,
                    "shots_assignments": shots_assignment,
                    "backend_size": len(all_backends),
                },
                "run":{
                    "exp_dir": EXP_DIR,
                    "qubit": circuit_qubits,
                }
            }
            
            metadata["run"]["circuit_conf"] = circuit_conf
            metadata["run"]["circuit_qubits"] = penny_circuit.num_wires
            metadata["run"]["circuit_qasm"] = circuit_qasm
            metadata["run"]["circuit"] = circuit_name
            metadata["config"]["circuits_names"].append(circuit_name)
            metadata["config"]["circuits"].append(circuit_qasm)
            
            for observable in ["Z"*circuit_qubits]:
                
                metadata["run"]["observable"] = observable
                
                perf_exp_val = compute_theoretical_exp_value(circuit_qasm, observable)
                
                metadata["run"]["perf_exp_val"] = perf_exp_val
                
                with open(os.path.join(PERF_EXP_VAL_DIR, f"{circuit_name}_{circuit_qubits}_{observable}.json"), "w") as f:
                    json.dump(metadata, f)
                
                no_cc_no_sw_res = run(
                                    circuit = circuit_qasm,
                                    shots_assignment = shots_assignment,
                                    observable_string = observable,
                                    provider_backend_couple = all_backends,
                                    cutting = False,
                                    shot_wise = False,
                                    metadata = metadata,
                                )
                
                
                with open(os.path.join(NO_CC_NO_SW_DIR, f"{circuit_name}_{circuit_qubits}_{observable}.json"), "w") as f:
                    json.dump(no_cc_no_sw_res, f)
                if INT_GIT:
                    os.system("git pull && git add ./exps* && git add *.log")
                    os.system(f"git commit -m 'Exp - C {circuit_qubits}'")
                    os.system("git push")
                    
                for current_backends in tqdm(powerset(all_backends), desc="Backends"):
                    
                    if len(current_backends) == 0 or len(current_backends) == 1:
                        # print("Skipping: ", current_backends, flush=True)
                        continue
                    
                    print("Current backends: ", current_backends, flush=True)
                    
                    metadata["run"]["backends_size"] = len(current_backends)
                    metadata["run"]["backends"] = current_backends
                    backends_hash = str(hash(str(current_backends)))
                
                    for cut_strategy in cut_strategies:
                        if cut_strategy == "pennylane":
                            for cut_dict in [{
                                                "max_free_wires": circuit_qubits-1,
                                                "min_free_wires": 2, 
                                                "num_fragments_probed":(2, circuit_qubits//2+1)
                                            }]:
                                
                                cut_tool = (cut_strategy, cut_dict)
                                
                                metadata["run"]["cut_strategy"] = cut_strategy
                                metadata["run"]["cut_dict"] = cut_dict
                                metadata["run"]["cut_tool"] = cut_tool
                                
                                
                                no_sw_res_filename = f"{circuit_name}_{circuit_qubits}_{observable}_{cut_strategy}_{cut_dict}.json"
                                
                                if not os.path.exists(os.path.join(NO_SW_DIR, no_sw_res_filename)):
                                    cc_res = run(
                                        circuit = circuit_qasm,
                                        shots_assignment = shots_assignment,
                                        observable_string = observable,
                                        cut_tool = cut_tool,
                                        provider_backend_couple = all_backends, #NOTICE: all_backends here
                                        cutting = True,
                                        shot_wise = False,
                                        metadata = metadata,
                                    )
                                    
                                    with open(os.path.join(NO_SW_DIR, no_sw_res_filename), "w") as f:
                                        json.dump(cc_res, f)
                                    if INT_GIT:
                                        os.system("git pull && git add ./exps* && git add *.log")
                                        os.system(f"git commit -m 'Exp CC - B {len(current_backends)} - C {circuit_qubits}'")
                                        os.system("git push")
                                        
                                    if "failed" in cc_res and cc_res["failed"]:
                                        print("Failed: ", no_sw_res_filename, flush=True)
                                        
                            
                                for split_policy in shotwise_policies:
                                    def split(backends, shots):
                                        coefficients = globals()[split_policy](backends)
                                        dispatch = {}
                                        for provider in coefficients:
                                            dispatch[provider] = {}
                                            for backend in coefficients[provider]:
                                                dispatch[provider][backend] = int(coefficients[provider][backend]*shots)
                                                
                                        tot_shots = sum([sum(dispatch[provider].values()) for provider in dispatch])
                                        diff_shots = tot_shots - shots
                                        
                                        if diff_shots > 0:
                                            for i in range(diff_shots):
                                                provider,backend = backends[i]
                                                dispatch[provider][backend] -= 1
                                        elif diff_shots < 0:
                                            for i in range(-diff_shots):
                                                provider,backend = backends[i]
                                                dispatch[provider][backend] += 1
                                                
                                        dispatch_ls = []
                                        for provider in dispatch:
                                            for backend in dispatch[provider]:
                                                dispatch_ls.append((provider,backend,dispatch[provider][backend]))
                                        return dispatch_ls, coefficients
                                        
                                        
                                    for merge_policy in [split_policy]: #TODO: now only "pure" combinations | before was shotwise_policies | now [split_policy]
                                        def merge(counts):
                                            backends = []
                                            for provider in counts:
                                                for backend in counts[provider]:
                                                    backends.append((provider,backend))
                                                    
                                            coefficients = globals()[merge_policy](backends)
                                            probs = {}
                                            for provider in counts:
                                                for backend in counts[provider]:
                                                    for fragment_id, fargment_counts in counts[provider][backend]:
                                                        if fragment_id not in probs:
                                                            probs[fragment_id] = {}
                                                        for state,count in fargment_counts.items():
                                                            if state not in probs[fragment_id]:
                                                                probs[fragment_id][state] = 0
                                                            probs[fragment_id][state] += count*coefficients[provider][backend]
                                                                
                                                            
                                            for fragment_id in probs:
                                                total = sum([probs[fragment_id][state] for state in probs[fragment_id]])
                                                for state in probs[fragment_id]:
                                                    probs[fragment_id][state] /= total
                                                            
                                            return probs, coefficients
                                        
                                        metadata["run"]["split_policy"] = split_policy
                                        metadata["run"]["merge_policy"] = merge_policy
                                        
                                        sw_res_filename = f"{len(current_backends)}_{circuit_name}_{circuit_qubits}_{observable}_{split_policy}_{merge_policy}_{backends_hash}.json"
                                        
                                        
                                        if not os.path.exists(os.path.join(NO_CC_DIR, sw_res_filename)):
                                            sw_res = run(
                                                circuit = circuit_qasm,
                                                shots_assignment = shots_assignment,
                                                observable_string = observable,
                                                provider_backend_couple = current_backends,
                                                split_func = split,
                                                merg_func = merge,
                                                cutting = False,
                                                shot_wise = True,
                                                metadata = metadata,
                                            )
                                            
                                            with open(os.path.join(NO_CC_DIR, sw_res_filename), "w") as f:
                                                json.dump(sw_res, f)
                                            if INT_GIT:
                                                os.system("git pull && git add ./exps* && git add *.log")
                                                os.system(f"git commit -m 'Exp SW - B {len(current_backends)} - C {circuit_qubits}'")
                                                os.system("git push")
                                        
                                        cc_sw_res = run(
                                            circuit = circuit_qasm,
                                            shots_assignment = shots_assignment,
                                            observable_string = observable,
                                            cut_tool = cut_tool,
                                            provider_backend_couple = current_backends,
                                            split_func = split,
                                            merg_func = merge,
                                            cutting = True,
                                            shot_wise = True,
                                            metadata = metadata,
                                        )
                                        
                                        with open(os.path.join(CC_SW_DIR, f"{len(current_backends)}_{circuit_name}_{circuit_qubits}_{observable}_{cut_strategy}_{cut_dict}_{split_policy}_{merge_policy}_{backends_hash}.json"), "w") as f:
                                            json.dump(cc_sw_res, f)
                    if INT_GIT:
                        os.system("git pull && git add ./exps* && git add *.log")
                        os.system(f"git commit -m 'Exp BOTH - B {len(current_backends)} - C {circuit_qubits}'")
                        os.system("git push")
                                            
                                            
        
    if action in ("all", "an"):
        
        if action == "all":
            exp_dir = EXP_DIR
        elif action == "an" and len(sys.argv) > 2:
            exp_dir = sys.argv[2]
        else:
            latest_exp = max([os.path.join(EXPS_DIR, d) for d in os.listdir(EXPS_DIR)], key=os.path.getmtime)
            exp_dir = latest_exp
            
        print("Analyzing: ", exp_dir)
        
        CC_SW_DIR = os.path.join(exp_dir, "cc_sw")
        NO_CC_DIR = os.path.join(exp_dir, "no_cc")
        NO_SW_DIR = os.path.join(exp_dir, "no_sw")
        NO_CC_NO_SW_DIR = os.path.join(exp_dir, "no_cc_no_sw")
        PERF_EXP_VAL_DIR = os.path.join(exp_dir, "perf_exp_val")
        
        data = []
        
        for cc_sw in os.listdir(CC_SW_DIR):
            if cc_sw == ".DS_Store":
                continue
            with open(os.path.join(CC_SW_DIR, cc_sw), "r") as f:
                try:
                    cc_sw_res = json.load(f)
                except:
                    print("Failed Loading: ", cc_sw, flush=True)
                    exit()
                
                if "failed" in cc_sw_res and cc_sw_res["failed"]:
                    print("Failed: ", cc_sw, flush=True)
                    continue
                
                results = cc_sw_res["results"]
                perf_exp_val = cc_sw_res["metadata"]["run"]["perf_exp_val"]
                error = abs(results - perf_exp_val)
                
                shots = shots_from_dispatch(cc_sw_res["intermediate_results"]["dispatch"])
                
                data.append({
                    "backends_size": cc_sw_res["metadata"]["run"]["backends_size"],
                    "circuit": cc_sw_res["metadata"]["run"]["circuit"],
                    "qasm": cc_sw_res["metadata"]["run"]["circuit_qasm"],
                    "qubit": cc_sw_res["metadata"]["run"]["qubit"],
                    "observable": cc_sw_res["metadata"]["run"]["observable"],
                    "backends": cc_sw_res["metadata"]["run"]["backends"],
                    "cut_strategy": cc_sw_res["metadata"]["run"]["cut_strategy"],
                    "cut_dict": cc_sw_res["metadata"]["run"]["cut_dict"],
                    "split_policy": cc_sw_res["metadata"]["run"]["split_policy"],
                    "merge_policy": cc_sw_res["metadata"]["run"]["merge_policy"],  
                    "shot_wise": True,
                    "circuit_cutting": True,
                    "expected_value": results,
                    "error": error,
                    "time_cutting": cc_sw_res["times"]["time_cutting"],
                    "time_dispatch": cc_sw_res["times"]["time_dispatch"],
                    "time_execution": cc_sw_res["times"]["time_execution"] if "time_execution" in cc_sw_res["times"] else None,
                    "time_simulation": cc_sw_res["times"]["time_simulation"] if "time_simulation" in cc_sw_res["times"] else None,
                    "time_merge": cc_sw_res["times"]["time_merge"],
                    "time_expected_values": cc_sw_res["times"]["time_expected_values"],
                    "time_sew": cc_sw_res["times"]["time_sew"],
                    "time_total": cc_sw_res["times"]["time_total"],
                    "dispatch": cc_sw_res["intermediate_results"]["dispatch"],
                    "total_shots": sum([sum(s) for s in shots]),
                    "avg_shots_per_backend": sum([sum(s) for s in shots])/len(shots),
                    "num_fragments": cc_sw_res["cut_info"]["num_fragments"],
                    "num_variations": cc_sw_res["cut_info"]["num_variations"],
                    "max_fragment_qubits": max(cc_sw_res["cut_info"]["fragments_qubits"])
                })
                
        for no_cc in os.listdir(NO_CC_DIR):
            if no_cc == ".DS_Store":
                continue
            with open(os.path.join(NO_CC_DIR, no_cc), "r") as f:
                try:
                    no_cc_res = json.load(f)
                except:
                    print("Failed Loading: ", no_cc, flush=True)
                    exit()
                
                if "failed" in no_cc_res and no_cc_res["failed"]:
                    print("Failed: ", no_cc, flush=True)
                    continue
                
                perf_exp_val = no_cc_res["metadata"]["run"]["perf_exp_val"]
                results = no_cc_res["results"]
                error = abs(results - perf_exp_val)
                
                shots = shots_from_dispatch(no_cc_res["intermediate_results"]["dispatch"])
                
                data.append({
                    "backends_size": no_cc_res["metadata"]["run"]["backends_size"],
                    "circuit": no_cc_res["metadata"]["run"]["circuit"],
                    "qasm": no_cc_res["metadata"]["run"]["circuit_qasm"],
                    "qubit": no_cc_res["metadata"]["run"]["qubit"],
                    "observable": no_cc_res["metadata"]["run"]["observable"],
                    "backends": no_cc_res["metadata"]["run"]["backends"],
                    "cut_strategy": None,
                    "cut_dict": None,
                    "split_policy": no_cc_res["metadata"]["run"]["split_policy"],
                    "merge_policy": no_cc_res["metadata"]["run"]["merge_policy"],
                    "shot_wise": True,
                    "circuit_cutting": False,
                    "expected_value": results,
                    "error": error,
                    "time_cutting": None,
                    "time_dispatch": no_cc_res["times"]["time_dispatch"],
                    "time_execution": no_cc_res["times"]["time_execution"] if "time_execution" in no_cc_res["times"] else None,
                    "time_simulation": no_cc_res["times"]["time_simulation"] if "time_simulation" in no_cc_res["times"] else None,
                    "time_merge": no_cc_res["times"]["time_merge"],
                    "time_expected_values": no_cc_res["times"]["time_expected_values"],
                    "time_sew": None,
                    "time_total": no_cc_res["times"]["time_total"],
                    "dispatch": no_cc_res["intermediate_results"]["dispatch"],
                    "total_shots": sum([sum(s) for s in shots]),
                    "avg_shots_per_backend": sum([sum(s) for s in shots])/len(shots),
                    "num_fragments": None,
                    "num_variations": None,
                    "max_fragment_qubits": None,
                })
                
        for no_sw in os.listdir(NO_SW_DIR):
            if no_sw == ".DS_Store":
                continue
            with open(os.path.join(NO_SW_DIR, no_sw), "r") as f:
                try:
                    no_sw_res = json.load(f)
                except:
                    print("Failed Loading: ", no_sw, flush=True)
                    exit()
                
                if "failed" in no_sw_res and no_sw_res["failed"]:
                    print("Failed: ", no_sw, flush=True)
                    continue
                
                perf_exp_val = no_sw_res["metadata"]["run"]["perf_exp_val"]
                results = no_sw_res["results"]
                
                for provider in results:
                    for backend in results[provider]:
                        exp_value = results[provider][backend]
                        
                        error = abs(exp_value - perf_exp_val)
                        
                        shots = shots_from_dispatch(no_sw_res["intermediate_results"]["dispatch"])
                
                        data.append({
                            "backends_size": 1,
                            "circuit": no_sw_res["metadata"]["run"]["circuit"],
                            "qasm": no_sw_res["metadata"]["run"]["circuit_qasm"],
                            "qubit": no_sw_res["metadata"]["run"]["qubit"],
                            "observable": no_sw_res["metadata"]["run"]["observable"],
                            "backends": [(provider, backend)],
                            "cut_strategy": no_sw_res["metadata"]["run"]["cut_strategy"],
                            "cut_dict": no_sw_res["metadata"]["run"]["cut_dict"],
                            "split_policy": None,
                            "merge_policy": None,
                            "shot_wise": False,
                            "circuit_cutting": True,
                            "expected_value": exp_value,
                            "error": error,
                            "time_cutting": no_sw_res["times"]["time_cutting"],
                            "time_dispatch": no_sw_res["times"]["time_dispatch"],
                            "time_execution": no_sw_res["times"]["time_execution"] if "time_execution" in no_sw_res["times"] else None,
                            "time_simulation": no_sw_res["times"]["time_simulation"] if "time_simulation" in no_sw_res["times"] else None,
                            "time_merge": None,
                            "time_expected_values": no_sw_res["times"]["time_expected_values"][provider][backend],
                            "time_sew": no_sw_res["times"]["time_sew"][provider][backend],
                            "time_total": no_sw_res["times"]["time_total"],
                            "dispatch": no_sw_res["intermediate_results"]["dispatch"],
                            "total_shots": sum(shots[0]),
                            "avg_shots_per_backend": sum(shots[0])/len(shots[0]),
                            "num_fragments": no_sw_res["cut_info"]["num_fragments"],
                            "num_variations": no_sw_res["cut_info"]["num_variations"],
                            "max_fragment_qubits": max(no_sw_res["cut_info"]["fragments_qubits"])
                        })
                        
        for no_cc_no_sw in os.listdir(NO_CC_NO_SW_DIR):
            if no_cc_no_sw == ".DS_Store":
                continue
            with open(os.path.join(NO_CC_NO_SW_DIR, no_cc_no_sw), "r") as f:
                try:
                    no_cc_no_sw_res = json.load(f)
                except:
                    print("Failed Loading: ", no_cc_no_sw, flush=True)
                    exit()
                
                if "failed" in no_cc_no_sw_res and no_cc_no_sw_res["failed"]:
                    print("Failed: ", no_cc_no_sw, flush=True)
                    continue
                
                perf_exp_val = no_cc_no_sw_res["metadata"]["run"]["perf_exp_val"]
                results = no_cc_no_sw_res["results"]
                
                for provider in results:
                    for backend in results[provider]:
                        exp_value = results[provider][backend][list(results[provider][backend].keys())[0]]
                        error = abs(exp_value - perf_exp_val)
                        
                        shots = shots_from_dispatch(no_cc_no_sw_res["intermediate_results"]["dispatch"])
                        
                
                        data.append({
                            "backends_size": 1,
                            "circuit": no_cc_no_sw_res["metadata"]["run"]["circuit"],
                            "qasm": no_cc_no_sw_res["metadata"]["run"]["circuit_qasm"],
                            "qubit": no_cc_no_sw_res["metadata"]["run"]["qubit"],
                            "observable": no_cc_no_sw_res["metadata"]["run"]["observable"],
                            "backends": [(provider, backend)],
                            "cut_strategy": None,
                            "cut_dict": None,
                            "split_policy": None,
                            "merge_policy": None,
                            "shot_wise": False,
                            "circuit_cutting": False,
                            "expected_value": exp_value,
                            "error": error,
                            "time_cutting": None,
                            "time_dispatch": no_cc_no_sw_res["times"]["time_dispatch"],
                            "time_execution": no_cc_no_sw_res["times"]["time_execution"] if "time_execution" in no_cc_no_sw_res["times"] else None,
                            "time_simulation": no_cc_no_sw_res["times"]["time_simulation"] if "time_simulation" in no_cc_no_sw_res["times"] else None,
                            "time_merge": None,
                            "time_expected_values": no_cc_no_sw_res["times"]["time_expected_values"][provider][backend],
                            "time_sew": None,
                            "time_total": no_cc_no_sw_res["times"]["time_total"],
                            "dispatch": no_cc_no_sw_res["intermediate_results"]["dispatch"],
                            "total_shots": sum(shots[0]),
                            "avg_shots_per_backend": sum(shots[0])/len(shots[0]),
                            "num_fragments": None,
                            "num_variations": None,
                            "max_fragment_qubits": None
                        })
        
        df = pd.DataFrame(data)
        df.to_csv(os.path.join(exp_dir, "results.csv"), index=False)
        

                    
                    