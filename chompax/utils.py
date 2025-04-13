import jax.numpy as jnp
from time import perf_counter as time
def construct_kinematic_chains(problem):
    class Node:
        def __init__(self, id, parent=None, children=None):
            self.id = id
            self.parent = parent
            self.children = children if children else []

        def is_leaf(self):
            return len(self.children) == 0

        def add_child(self, child):
            self.children.append(child)

        def update_parent(self, parent):
            self.parent = parent

        def add_children(self, children):
            self.children.extend(children)

            for child in children:
                child.update_parent(self)

        def get_next(self, child):
            # get the next child. If there is no next child, check the parent
            if child is None:
                return self.children[0]
            else:
                idx = self.children.index(child)
                if idx + 1 < len(self.children):
                    return self.children[idx + 1]
                else:
                    # check the parent
                    if self.parent is None:
                        return None
                    else:
                        return self.parent.get_next(self)

        def __repr__(self):
            return str(self.id)

    # Prepare the nodes
    first_node = Node(problem[0])
    first_inspection = problem[1] if isinstance(problem[1], list) else [problem[1]]
    first_node.add_children([Node(item) for item in first_inspection])

    nodes = [first_node, *first_node.children]
    leafes = []
    current_node = first_node.get_next(None)  # get the first child
    i = 2
    while True:
        elements_to_be_inspected = (
            problem[i] if isinstance(problem[i], list) else [problem[i]]
        )
        if -1 in elements_to_be_inspected:  # we have reached the end of a chain
            leafes.append(current_node)
            # get next sibling. If there is none, check the parent too. Terminates if there is no parent
            current_node = current_node.parent.get_next(current_node)
            if current_node is None:
                break
        else:
            childs = [Node(item) for item in elements_to_be_inspected]
            nodes.extend(childs)
            current_node.add_children(childs)
            # if len(childs) > 1: #we have a branch
            current_node = childs[0]  # start with the first child
        i += 1
        if i >= len(problem):
            break

    # backtrace the nodes to get the chains
    chains = []
    for leaf in leafes:
        chain = [leaf]
        while not chain[-1].parent is None:
            chain.append(chain[-1].parent)
        chain = chain[::-1]
        chains.append(chain)
    return [[node.id for node in chain] for chain in chains]

class JAXTimer:
    def __init__(self, n=1):
        self.n = n
        self.times = jnp.ones((n,)) * jnp.nan

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.times = jnp.array(self.times)
        mean = jnp.mean(self.times)
        median = jnp.median(self.times)
        best = jnp.min(self.times)
        worst = jnp.max(self.times)
        #find best resolution for printing
        if mean >= 1:
            unit = "s"
            scale = 1
        elif mean >= 1e-3:
            unit = "ms"
            scale = 1e3
        elif mean >= 1e-6:
            unit = "us"
            scale = 1e6
        else:
            unit = "ns"
            scale = 1e9
        print(f"Mean: {mean*scale:.2f} {unit}, Median: {median*scale:.2f} {unit}, Best: {best*scale:.2f} {unit}, Worst: {worst*scale:.2f} {unit}")

    def time_function(self, func, *args, **kwargs):
        result = None
        for i in range(self.n):
            self.start_time = time()
            result = func(*args, **kwargs)
            result.block_until_ready()
            self.end_time = time()
            self.times = self.times.at[i].set(self.end_time - self.start_time)
        return result

def normalize_0_1(values, val_min, val_max):
    #normalize the values, that are in range (val_min, val_max) to be within (0,1)
    return (values - val_min) / (val_max - val_min)

def linear_schedule(start_value, end_value, num_steps):
    """
    Creates a linearly decreasing schedule of values.
    Args:
        max_val (float): The maximum value at the start of the schedule.
        min_val (float): The minimum value at the end of the schedule.
        num_steps (int): The number of steps over which to interpolate.
    Returns:
        Array: Values linearly interpolated between `max_val` and `min_val` for each step.
    """
    def fun(i): return (start_value - end_value) * \
        (1 - i / num_steps) + end_value
    return jnp.array([fun(i) for i in range(num_steps)])
