# Reducer Architecture in Haskell

For many years now I've been working with reducer architectures such as
[Redux](https://redux.js.org/) and [Elm](https://elm-lang.org/). I have
implemented several production solutions using these patterns in C++,
Objective-C, Swift, Kotlin, and F#. I have also worked directly with the
implementors of some of the open source solutions in this space.

Frameworks applying these patterns build upon a single common pattern
with variations for different purposes and goals. The properties that
most interest me are the compositionality of the approach and separation
of side effects.

I'll explore implementing the foundations of such a solution in Haskell.
I won't implement a full solution here. This primarily will capture some
of my think in the form of notes, and hopefully it is useful for some
readers. It is not a tutorial; if reading it as such, it may be helpful
to skim to the end and then work your way back to fill in gaps.

This is a literate Haskell file. First we'll need some preliminaries,
only a few language extensions and imports. We depend on the
[mtl](https://hackage.haskell.org/package/mtl) and
[lens](https://hackage.haskell.org/package/lens) packages.

```haskell
{-# LANGUAGE DeriveFunctor #-}
{-# LANGUAGE RankNTypes #-}
{-# LANGUAGE TemplateHaskell #-}

module Main where

import Control.Applicative (liftA2)
import Control.Lens
import Control.Monad.State (State, get, modify, put, runState)
import Data.Foldable (traverse_)
import Data.Function (fix)
import Data.IORef (newIORef, readIORef, writeIORef)
import qualified Data.HashMap as M
import Data.UUID (UUID)
import Data.UUID.V4 (nextRandom)
import Data.Void (Void, absurd)
```

## Sink

A callback, in general, is pretty simple. We use them all the time, for
example, in UI code. Modeling our particular callback in Haskell is
pretty interesting because it exposes how much implicit behavior we are
allowing in other languages. The `Sink` type (as in "event sink") helps
describe the shape of a how we send actions and is used to construct our
`Effect` type's callback.

Noting the implicit side-effects, the Elmish equivalent is:

     type Dispatch<'msg> = 'msg -> unit

Swift might define this as:

     typealias Sink<A> = (A) -> Void

In Haskell we must run this in `IO` to account for side-effects.
Normally we would not call this out as its own type alias, but it can be
useful for comparing with other frameworks.

```haskell
type Sink a = a -> IO ()
```

## Effect

In [Elmish](https://elmish.github.io/elmish/) (F#), the Effect is called
`Sub` and has an equivalent signature:

     type Sub<'msg> = Dispatch<'msg> -> unit

Note that Elm proper (not Elmish) does not describe these types as
functions. It defines them as reified values which must be transported
across the Elm / JavaScript boundary to be interpreted. Spotify's
[Mobius](https://github.com/spotify/mobius) retains this reified effect
approach and therefore must implement an
[EffectHandler](https://github.com/spotify/mobius/wiki/Effect-Handler)
for interpreting the effects, a distinctly non-compositional approach
that requires significant boilerplate.

In Swift, such as in [The Composable
Architecture](https://github.com/pointfreeco/swift-composable-architecture)
by [Point-Free](https://pointfree.co/), a basic effect type can be
described as follows.

     struct Effect<A> {
         let run: (@escaping (A) -> Void) -> Void
     }

Note that `Effect a` is also equivalent to [Rx
observables](http://reactivex.io/documentation/observable.html) (and
[Combine
Publishers](https://developer.apple.com/documentation/combine/publisher)
in Swift). Rather than implementing lots of operations in terms of
`Effect a` (e.g., debounce, zip, schedulers, etc.), you can consider
relying on a reactive framework's `Observable` implementation. To do
this, simply provide a function that can wrap an `Observable` in an
`Effect`. This is how Point-Free's Composable Architecture works and how
Elmish recommends you implement certain complicated effects.

I have also used Rx directly (i.e., where `Effect` is exactly an
`Observable`) as well as coroutines in Kotlin implementations of the
reducer architecture. It ends up looking very similar to other
approaches, though it is preferable to use a more narrowly defined types
within the reducer framework.

An effect is a continuation-like type. It has the basic shape of:

     (a -> r) -> r

`r` in this case is side-effectful and carries no data, so we can
rewrite that as for Haskell (for `r == IO ()` and noting that
`a -> IO ()` is our `Sink` type):

     (a -> IO ()) -> IO ()

```haskell
newtype Effect a = Effect { runEffect :: Sink a -> IO () }
    deriving Functor
```

A `Semigroup` instance combines two effects into one. Note that this is
different than the semantics of the `Applicative` instance for `ContT`.
We can't rely on `ContT` for sequencing since it will drop all but the
first callback invocation.

```haskell
instance Semigroup (Effect a) where
    Effect x <> Effect y = Effect $ \cb ->
        x cb >> y cb
```

The `Monoid` instance allows to declare an empty effect type.

```haskell
instance Monoid (Effect a) where
    mempty = Effect . const $ pure ()
```

The `Applicative` instance allows us to lift values into the effect,
when we don't want to perform a side-effect other than to inject a new
action into the reducer. The implementation of this instance is
identical to that for the `Cont` monad.

```haskell
instance Applicative Effect where
    pure = Effect . flip id
    Effect ab <*> Effect a = Effect $ \b -> ab (\f -> a (b . f))
```

A "void" effect never calls its callback. Not that we make this
polymorphic in its returned `Effect`, eliminating the need to write
`fmap absurd` within the reducer logic.

```haskell
voidEffect :: IO () -> Effect a
voidEffect = fmap absurd . Effect . const
```

## Reducer

Generally speaking a reducer is a function *S×A → S* (i.e.,
`s -> a -> s`). We will decorate this shape with 1) first class effects,
and 2) an enviroment that contains all our side-effect domain.

In Elmish this would be:

     type Cmd<'msg> = Sub<'msg> list

     update : 'msg -> 'model -> 'model * Cmd<'msg>

Where `Cmd<'msg>` is just a list of `Sub<'msg>`. We don't do this
because `Effect` is a monoid. List is the free monoid, which is
equivalent, but less ergonomic for some cases. Elm uses the list as a
free monoid presumably because it does not support higher kinded types.

The Composable Architecture describes a reducer as follows, with an
additional Environment type that carries the side effect
implementations.

     struct Reducer<State,Action,Environment> {
         let reduce: (inout State, Action, Environment) -> Effect<A>
     }

`inout` in Swift is like a mutable borrow in Rust or a State monad. It
is a hygienic mutation, not a leakable reference that can be escaped via
a closure. In other words, it is fully equivalent to:

     (State, Action, Environment) -> (State, Effect<Action>)

Another way to describe how we decorate the reducer is that if a
`Reducer` is some function (renaming `s` and `a` to avoid term conflict
with the subsequent definition):

     f : x -> y -> x

Then we would define our decorated reducer as:

     f : (s, Effect a) -> (a, r) -> (s, Effect a)

In other words, `x = (s, Effect a)` and `y = (a, r)`. By convention, we
would typically call this function with the current state and an empty
effect. We could also thread through accumulating events here, however.
That might be worth exploring, but increases the risk that the reducer
implementer accidentally forget to propagate effects.

In Haskell, the reducer looks similar to the [`Update`
monad](https://chrispenner.ca/posts/update-monad) (possibly the origin
of the Elm naming reducers "update" instead of "reducer" or "fold"). The
`Update` monad has a monoid requirement on the action type, however:

     class (Monoid p) => ApplyAction p s where
         applyAction :: p -> s -> s

This constraint requires us to place the action (in our case `(a, r)`)
in a list (the free monoid). In general, the ergonomics and performance
of the `Update` monad won't help us.

We could also use
[`Reducer c m`](https://hackage.haskell.org/package/reducers-3.12.3/docs/Data-Semigroup-Reducer.html)
from `Data.Semigroups.Reducer`, but that would require a monoidal
wrapper type on the product type `(s, eff)`. That's definitely possible,
but requires a strict notion of an empty type for `s`. For some types, a
`Monoid` instance may already be defined and may not be equal to the
initial state, presenting a slightly awkward API.

If you want to stay monoidal in `m`, then your semigroup combination
rule for `m` would simply be:

     (_, effA) <> (s , effB) = (s, effA <> effB)

We drop the old state and only preserve the newer state. However, we
need to accumulate the effects. The equivalence to the `Writer` monad
should be clear here, since, like the `Writer` monad, we are dealing
with a tuple of some computed value and monoidal right-hand term.

In our implementation, we use the `State` monad for tracking state
changes. We also define a simple `Semigroup` instance for the Reducer
type which merges the effects, and a `Monoid` instance which allows us
to produce an empty reducer.

More work could be done here to make reducer composition easier. Reducer
is both a contravariant and covariant functor in `a`. These facts can be
used to define `Applicative` and `Monad` instances for Reducer. That
might expose some ergonomic, architectural, or perfomance improvements
for pullbacks.

```haskell
newtype Reducer s a r
  = Reducer { runReducer :: a -> r -> State s (Effect a) }
```

We can combine two reducers by running them sequentially and combining
their effects.

```haskell
instance Semigroup (Reducer s a r) where
    x <> y = Reducer $ \a r ->
        liftA2 (<>) (runReducer x a r) (runReducer y a r)
```

A reducer can be empty, meaning it does not modify its state and it
returns no effects. For that we define a helper for an empty effect.

```haskell
noEffect :: State s (Effect a)
noEffect = pure mempty
```

We use it within the Monoid instance for Reducer.

```haskell
instance Monoid (Reducer s a r) where
    mempty = Reducer $ \_ _ -> noEffect
```

## Store

Mutation and side-effect execution is isolated to the dispatch function.
Elmish and Elm expose a `Program` notion instead of a `Store` (because
they have no notion of pullbacks for reducer composition).

Swift:

     struct Store<S,A> {
         let send: (A) -> Void
         let subscribe: ((S) -> Void) -> Void

         init(
             initialState: S,
             reducer: Reducer<S,A>
         ) { /* ... */ }
     }

In Haskell we run inside of the `State` and `IO` monads. This code is
fairly imperative--perhaps there are more clever approaches--but it gets
the job done.

```haskell
mkStore
    :: Reducer s a r
    -> s
    -> r
    -> IO (Sink a, Sink s -> IO (IO ()))
mkStore r initialState env = do
    stateRef <- newIORef initialState
    subRef <- newIORef M.empty

    -- We return a recursive `Sink` function
    let dispatch = fix (\send a -> do
         -- Get the current state
          s <- readIORef stateRef
          -- Run the provided reducer in the State monad
          let (Effect cont, s') = runState (runReducer r a env) s
          -- Update our reference to point to the new state
          writeIORef stateRef s'
          -- Notify all subscribers of new state
          readIORef subRef >>= traverse_ ($ s')
          -- Invoke the side-effect, pointing back to ourself
          cont send)

    let subscribe notify = do
          -- Notify subscriber of current state
          readIORef stateRef >>= notify
          -- Get the current list of subscribers
          subs <- readIORef subRef
          -- Generate a new identifier for later cancellation
          uuid <- nextRandom
          -- Update list of subscribers
          writeIORef subRef (M.insert uuid notify subs)
          -- Return an IO action to unsubscribe
          pure $ do
              -- Get the current list of subscribers
              subs <- readIORef subRef
              -- Update the list of subscribers after deleting current one
              writeIORef subRef (M.delete uuid subs)

    pure (dispatch, subscribe)
```

Here we're creating a mutable variable in Haskell, just like we would
normally use in a non-pure language. This is viable in a garbage
collected language, and the mutation is hygienically isolated to this
function. In the Haskell world this happens to be the same strategy
employed by [miso](https://github.com/dmjio/miso).

The alternative is to run the loop in the state monad. That approach
works but gives the effect implementation thunk direct access to the
state, a violation of the consistency requirement of state updates going
exclusively through the reducer.

## Pullback

The pullback is a lifting operation that allows a reducer with some
"smaller" domain to run over a "larger" domain. To do this we need to
provide a lot of transformation functions. This function represents the
largest learning curve of the reducer architecture. It is also the magic
piece that enables app wide compositional state sharing. It is best
understood by trying to implement the function
`Reducer s a e -> ??? -> Reducer t b f` on your own.

The implementation here follows the techniques from The Composable
Architecture. There are many pullback implementations possible for
different domains. This simple pullback is what is used when your state
is a basic product type and your action space is a sum type.

There are a lot of type variables here:

-   `s` - Local state (e.g., `CounterState`)
-   `t` - Global state (e.g., `AppState`, which contains a
    `CounterState`)
-   `a` - Local action (e.g., a possible sub-action of `AppAction`)
-   `b` - Global action (e.g., `AppAction`)
-   `e` - The local environment
-   `f` - The global environment (from which we can produce a local env)

And we are pulling back over a number of axes:

1)  *State*: We need to provide a way to extract the local state from
    the global state, and to embed a new local state into an updated
    global state. That is a pair of functions `(t -> s, t -> s -> t)`,
    or equivalently a `Lens' t s`.

2)  *Action*: We need a way to construct a global action from a local
    action. We also need a way to extract our (optional) local action
    from the global action. The extraction can fail because the action
    may not be targeted at this local domain. This is a pair of
    functions `(a -> b, b -> Maybe a)`, or equivalently a `Prism' b a`.

3)  *Environment*: We need a way to get a more narrow local environment
    out of a parent global environment. This is simply a function
    `f -> e`.

This code uses optics from the
[lens](https://hackage.haskell.org/package/lens) package, which is
rather bewildering to the uninitiated.

In Elm, Redux, and other frameworks, this type of composition is not
supported. For example, in Elm, you would simply call subsidiary update
functions manually inside your reducer. That introduces a tight coupling
at the business logic layer between the various reducers and does not
permit modular testing and a host of other state sharing advantages.

```haskell
pullback
    :: Reducer s a e    -- local reducer
    -> Lens' t s        -- state mapping (t=global, s=local)
    -> Prism' b a       -- action mapping (b=global, a=local)
    -> (f -> e)         -- environment mapping (f=global, e=local)
    -> Reducer t b f    -- global reducer
pullback r state action env =
    -- Build a new reducer that operates over global state and action space.
    Reducer $ \b e ->
        -- Try to extract the local action from the global action
        case b ^? action of
            -- This action was not for us, so ignore it.
            Nothing -> noEffect
            -- This action is for us, so now we have extracted a local action.
            Just a -> do
                t <- get
                -- Run extract the local state and environment so we can run the
                -- local reducer, getting a new local state and an effect in the
                -- local action space.
                let (eff, s) = runState (runReducer r a $ env e) (t ^. state)
                -- Construct a new global state with the new local state we got.
                put $ t & state .~ s
                -- Map the effect into the global action space.
                pure $ review action <$> eff
```

That is the end of any framework code. The rest of this file is example
usage.

## Example: Print Effect

This is an example of a simple effect. We'll throw this in a modularized
environment. This effect has a type `Void` since it doesn't invoke its
callback. It therefore also ignores the callback parameter.

```haskell
printToConsole :: String -> Effect a
printToConsole = voidEffect . putStrLn
```

## Example: Higher Order Reducer

This function takes a reducer as an argument and decorates its behavior.
In this case we need to be told how to extract the print effect from the
environment.

```haskell
logging
    :: (Show s, Show a)           -- Need these because we're printing things
    => Reducer s a r              -- the reducer to be decorated
    -> (r -> String -> Effect a)  -- the function from env to print effect
    -> Reducer s a r              -- decorated reducer of same type
logging r logStr = Reducer $ \a env -> do
    -- Run the original reducer, remembering the original state, new state, and
    -- effect.
    s0  <- get
    eff <- runReducer r a env
    s1  <- get

    -- Logging details for the effects.
    let actionStr = "LOG: ACTION: " ++ show a
        stateStr  = "LOG: STATE: " ++ show s0 ++ " -> " ++ show s1
        writeLog  = logStr env

    -- Return the combined effects, prepending our logging effects
    pure $ (writeLog actionStr <> writeLog stateStr) <> eff
```

## Example: Counter

Importantly, this example is self-contained and modularized. It has no
dependencies on other example values such as `AppState` or
`printToConsole` and could be compiled and tested separately from the
rest of the app.

Example environment. In this case, we don't need a wrapper type since
our environment is just a print/logging function. It will be provided by
the parent scope during pullback or it could be mocked during test.

```haskell
type CounterEnv = String -> Effect CounterAction
```

We the domain of this example is a an integer state and a simple action
space. It can only increment/decrement the counter state.

```haskell
data CounterAction
    = Incr
    | Decr
    deriving Show
```

This reducer is a little contrived. It will always increment when
requested. When a decrement is requested, it will complain if the state
is already zero. If it is greater than zero, then it will continuously
decrement until it is zero. This is accomplished by feeding back `Decr`
actions.

```haskell
counter :: Reducer Int CounterAction CounterEnv
counter = Reducer $ \a writeLine ->
    case a of
    Incr -> modify (+1) >> noEffect
    Decr -> do
        s <- get
        if s == 0
            then pure $ writeLine "at zero!"
            else do
                put (s-1)
                pure $
                  writeLine ("plunge from " ++ show s)
                  <> pure Decr
```

## Example: Application

Example larger state which manages two counters.

```haskell
data AppState
    = AppState
    { _counter1 :: Int
    , _counter2 :: Int
    }

makeLenses ''AppState
```

Custom `Show` instance to make the program output a little tidier.

```haskell
instance Show AppState where
    show s = "(" <> show (s^.counter1) <> ", " <> show (s^.counter2) <> ")"
```

Imagine our environment also contains lots of other effects here.

```haskell
newtype AppEnv
  = AppEnv
  { logLine :: forall a. String -> Effect a
  }
```

And an action domain for our application.

```haskell
data AppAction
    = Counter1 CounterAction
    | Counter2 CounterAction
    deriving (Show)

makePrisms ''AppAction
```

Let's just imagine that it does other non-counter stuff

```haskell
appReducer :: Reducer AppState AppAction AppEnv
appReducer = mempty
```

## Example: Composition

This is our end goal, a composed global reducer which invokes logic of
subsidiary reducers.

```haskell
mainReducer :: Reducer AppState AppAction AppEnv
mainReducer
    = appReducer
    <> pullback counter counter1 _Counter1 logLine
    <> pullback counter counter2 _Counter2 logLine
```

## Example: Usage

This example function should be imagined to be something like a UI where
the user is tapping on + and - buttons.

```haskell
script :: [AppAction]
script = [
    Counter1 Incr,    -- state = (1,0)
    Counter1 Incr,    -- state = (2,0)
    Counter1 Incr,    -- state = (3,0)
    Counter1 Decr,    -- state = (2,0), (1,0), (0, 0) + prints "at zero!
    Counter1 Decr,    -- state = (0,0) + prints "at zero!"
    Counter2 Incr,    -- state = (0,1)
    Counter2 Decr,    -- state = (0,0)
    Counter2 Decr ]   -- state = (0,0) + prints "at zero!"
```

```haskell
example :: Sink AppAction -> IO ()
example send = traverse_ send script
```

Kick things off!

```haskell
main :: IO ()
main = do
    -- Establish the initial state (both counters are 0)
    let initialState = AppState { _counter1 = 0, _counter2 = 0 }

    -- Establish our environment. If testing, we could be providing mocks.
    let env = AppEnv { logLine = printToConsole }

    -- Establish our reducer. In this case we are decorating with logging effects.
    let r = logging mainReducer logLine

    -- Construct a dispatch function for dispatching actions. This is equivalent
    -- to a `Store` in redux.
    (send, subscribe) <- mkStore r initialState env
    unsubscribe <- subscribe $ \s -> putStrLn $ "OBSERVED: " ++ show s
    example send
    unsubscribe
```

Produces the following output when run:

    OBSERVED: (0, 0)
    OBSERVED: (1, 0)
    LOG: ACTION: Counter1 Incr
    LOG: STATE: (0, 0) -> (1, 0)
    OBSERVED: (2, 0)
    LOG: ACTION: Counter1 Incr
    LOG: STATE: (1, 0) -> (2, 0)
    OBSERVED: (3, 0)
    LOG: ACTION: Counter1 Incr
    LOG: STATE: (2, 0) -> (3, 0)
    OBSERVED: (2, 0)
    LOG: ACTION: Counter1 Decr
    LOG: STATE: (3, 0) -> (2, 0)
    plunge from 3
    OBSERVED: (1, 0)
    LOG: ACTION: Counter1 Decr
    LOG: STATE: (2, 0) -> (1, 0)
    plunge from 2
    OBSERVED: (0, 0)
    LOG: ACTION: Counter1 Decr
    LOG: STATE: (1, 0) -> (0, 0)
    plunge from 1
    OBSERVED: (0, 0)
    LOG: ACTION: Counter1 Decr
    LOG: STATE: (0, 0) -> (0, 0)
    at zero!
    OBSERVED: (0, 0)
    LOG: ACTION: Counter1 Decr
    LOG: STATE: (0, 0) -> (0, 0)
    at zero!
    OBSERVED: (0, 1)
    LOG: ACTION: Counter2 Incr
    LOG: STATE: (0, 0) -> (0, 1)
    OBSERVED: (0, 0)
    LOG: ACTION: Counter2 Decr
    LOG: STATE: (0, 1) -> (0, 0)
    plunge from 1
    OBSERVED: (0, 0)
    LOG: ACTION: Counter2 Decr
    LOG: STATE: (0, 0) -> (0, 0)
    at zero!
    OBSERVED: (0, 0)
    LOG: ACTION: Counter2 Decr
    LOG: STATE: (0, 0) -> (0, 0)
    at zero!
