from __future__ import annotations

from abc import ABC, abstractmethod

# Sometimes, when the KeyError is raised, it might become a problem. 
# To overcome this Python introduces another dictionary like container known as Defaultdict which is present inside the collections module.
# d = defaultdict(lambda: "Not Present") will  return "Not Present" when key being queried is absent instead of giving key error 
# dict2 = defaultdict(float) will return 0.0 float when the queried key value pair is absent 
# d["a"] = 1 ,,,, dict2['a'] = 1
# d["b"] = 2 ,,,, dict2['b'] = 2
# print(d["a"]) => 1  ,,,, print(dict2["a"]) => 1 
# print(d["b"]) => 2  ,,,, print(dict2["b"]) => 2 
# print(d["c"]) => "Not Present" ,,,, print(dict2["c"]) => 0.0

from collections import defaultdict
import numpy as np
import random

from typing import (Callable, Dict, Generic, Iterator,
                    Mapping, Set, Sequence, Tuple, TypeVar)

A = TypeVar('A')

B = TypeVar('B')

# create a class distribution thats inherits an abstract base class (2.15.9 in notes) , 
# and a generic class that just tells that the variable used should be be of type A (you can remove it if you want , no real advantage apart from type hinting )
class Distribution(ABC, Generic[A]):

    # A probability distribution that we can sample.
    
    # Abstract Method/function that will return a random value from the probability distribution  
    @abstractmethod
    def sample(self) -> A:
        '''Return a random sample from this distribution.
        '''
        pass
    # will return a list of N values from the probability distribution , by calling self.sample multiple times  
    def sample_n(self, n: int) -> Sequence[A]:
        '''Return n samples from this distribution.'''
        return [self.sample() for _ in range(n)]

    # Abstract Method/function that will return the expecation, ormean of values that f(X) can take where X is the random variable for the distribution 
    # and f is an arbitrary function from X to float
    # Callable[[A], float] means , function f takes one input arguement of type A, and returns a float 
    @abstractmethod
    def expectation(self,f: Callable[[A], float]) -> float:
        '''Return the expecation of f(X) where X is the
        random variable for the distribution and f is an
        arbitrary function from X to float
        '''
        pass

    # Function that will return the object of the class SampleDistribution (defined next), 
    # instantiated with the function f ( Callable[[A], B] ), (meaning that the function takes in an arguement of type A, and returns a value of type B )
    def map( self , f: Callable[[A], B] ) -> Distribution[B]:
        '''Apply a function to the outcomes of this distribution.'''
        # and return an object of the class SampleDistribution instantiated with that function 
        return SampledDistribution(lambda: f(self.sample()))

    # Function that will return the object of the class SampleDistribution (defined next), 
    # instantiated with the function sample defined within
    def apply( self , f: Callable[[A], Distribution[B]] ) -> Distribution[B]:
        '''Apply a function that returns a distribution to the outcomes of
        this distribution. This lets us express *dependent random
        variables*.
        '''
        def sample():
            # self.sample() gets redefined in the class that takes Distribution as its abstract base class
            # (eg. SampledDistribution and self.sample there returns a call to the sampler function, which returns 
            # a random value from the probability distribution   
            a = self.sample()
            b_dist = f(a)
            return b_dist.sample()

        return SampledDistribution(sample)


# Another class SampledDistribution, that inherits and builds on the abstract Class Distribution
class SampledDistribution(Distribution[A]):
    '''A distribution defined by a function to sample it.
    '''
    sampler: Callable[[], A]
    expectation_samples: int

    def __init__( self , sampler: Callable[[], A] , expectation_samples: int = 10000 ):
        self.sampler = sampler
        self.expectation_samples = expectation_samples

    # return the sampler function defined in constructor __init__
    def sample(self) -> A:
        return self.sampler()

    #  call sampler function expectation samples number of times ( 10000 by default ) and then sum all those , 
    # values and divide by expectation sample (10000) , returning the average of those values 
    def expectation( self , f: Callable[[A], float] ) -> float:
        '''Return a sampled approximation of the expectation of f(X) for some f.
        '''
        return sum( f(self.sample()) for _ in range(self.expectation_samples) ) / self.expectation_samples

# A class that inherits the class SampledDistribution with all its methods and instantiates it with a uniform distribution
# (PG - 2.16.1) 
class Uniform(SampledDistribution[float]):
    '''Sample a uniform float between 0 and 1.
    '''
    def __init__(self, expectation_samples: int = 10000):
        # Arguements to instantiate the Super Class( Sampled distribution ), with a sampler,
        # that returns a random value from the uniform distribution (0-1)
        super().__init__(
            sampler=lambda: random.uniform(0, 1),
            expectation_samples=expectation_samples
        )

# A class that inherits the class SampledDistribution with all its methods and instantiates it with a Poissons distribution
# (PG - 2.16.5) 
class Poisson(SampledDistribution[int]):
    '''A poisson distribution with the given parameter.
    '''

    λ: float

    def __init__(self, λ: float, expectation_samples: int = 10000):
        self.λ = λ
        super().__init__(
            sampler=lambda: np.random.poisson(lam=self.λ),
            expectation_samples=expectation_samples
        )

# A class that inherits the class SampledDistribution with all its methods and instantiates it with a Poissons distribution
# (PG - 2.16.6) 
class Gaussian(SampledDistribution[float]):
    '''A Gaussian distribution with the given μ and σ.'''

    μ: float
    σ: float

    def __init__(self, μ: float, σ: float, expectation_samples: int = 10000):
        self.μ = μ
        self.σ = σ
        super().__init__(
            sampler=lambda: np.random.normal(loc=self.μ, scale=self.σ),
            expectation_samples=expectation_samples
        )


# An abstract class with the abstract method "table" that needs to implemented in all of the sub classes 
class FiniteDistribution(Distribution[A], ABC):
    '''A probability distribution with a finite number of outcomes, which
    means we can render it as a PDF or CDF table.
    '''
    @abstractmethod
    def table(self) -> Mapping[A, float]:
        '''Returns a tabular representation of the probability density
        function (PDF) for this distribution.
        '''
        pass

    def probability(self, outcome: A) -> float:
        '''Returns the probability of the given outcome according to this
        distribution.
        '''
        return self.table()[outcome]

    def map(self, f: Callable[[A], B]) -> FiniteDistribution[B]:
        '''Return a new distribution that is the result of applying a function
        to each element of this distribution.
        '''
        result: Dict[B, float] = defaultdict(float)

        for x, p in self:
            result[f(x)] += p

        return Categorical(result)

    def sample(self) -> A:
        outcomes = list(self.table().keys())
        weights = list(self.table().values())
        return random.choices(outcomes, weights=weights)[0]

    # TODO: Can we get rid of f or make it optional? Right now, I
    # don't think that's possible with mypy.
    def expectation(self, f: Callable[[A], float]) -> float:
        '''Calculate the expected value of the distribution, using the given
        function to turn the outcomes into numbers.
        '''
        return sum(p * f(x) for x, p in self)

    def __iter__(self) -> Iterator[Tuple[A, float]]:
        return iter(self.table().items())

    def __eq__(self, other: object) -> bool:
        if isinstance(other, FiniteDistribution):
            return self.table() == other.table()
        else:
            return False

    def __repr__(self) -> str:
        return repr(self.table())


# TODO: Rename?
class Constant(FiniteDistribution[A]):
    '''A distribution that has a single outcome with probability 1.
    '''
    value: A

    def __init__(self, value: A):
        self.value = value

    def sample(self) -> A:
        return self.value

    def table(self) -> Mapping[A, float]:
        return {self.value: 1}

    def probability(self, outcome: A) -> float:
        return 1. if outcome == self.value else 0.


class Bernoulli(FiniteDistribution[bool]):
    '''A distribution with two outcomes. Returns True with probability p
    and False with probability 1 - p.
    '''

    def __init__(self, p: float):
        self.p = p

    def sample(self) -> bool:
        return random.uniform(0, 1) <= self.p

    def table(self) -> Mapping[bool, float]:
        return {True: self.p, False: 1 - self.p}

    def probability(self, outcome: bool) -> float:
        return self.p if outcome else 1 - self.p


class Choose(FiniteDistribution[A]):
    '''Select an element of the given list uniformly at random.
    '''

    options: Set[A]

    def __init__(self, options: Set[A]):
        self.options = options

    def sample(self) -> A:
        return random.choice(list(self.options))

    def table(self) -> Mapping[A, float]:
        length = len(self.options)
        return {x: 1.0 / length for x in self.options}

    def probability(self, outcome: A) -> float:
        p = 1.0 / len(self.options)
        return p if outcome in self.options else 0.0


class Categorical(FiniteDistribution[A]):
    '''Select from a finite set of outcomes with the specified
    probabilities.
    '''

    probabilities: Mapping[A, float]

    def __init__(self, distribution: Mapping[A, float]):
        total = sum(distribution.values())
        # Normalize probabilities to sum to 1
        self.probabilities = {outcome: probability / total
                              for outcome, probability in distribution.items()}

    def table(self) -> Mapping[A, float]:
        return self.probabilities

    def probability(self, outcome: A) -> float:
        return self.probabilities.get(outcome, 0.)