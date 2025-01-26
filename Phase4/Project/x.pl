% Dynamic predicates for game state
:- dynamic bird_pos/2.
:- dynamic pig_pos/2.
:- dynamic rock_pos/2.

% Define the grid size
grid_size(8).

% State management
set_bird(X, Y) :-
    retractall(bird_pos(_, _)),
    assertz(bird_pos(X, Y)).

add_pig(X, Y) :-
    \+ pig_pos(X, Y),
    assertz(pig_pos(X, Y)).

add_rock(X, Y) :-
    \+ rock_pos(X, Y),
    assertz(rock_pos(X, Y)).

% Movement definition
move(up,    X, Y, X2, Y) :- X2 is X - 1, X2 >= 0.
move(down,  X, Y, X2, Y) :- grid_size(S), X2 is X + 1, X2 < S.
move(left,  X, Y, X, Y2) :- Y2 is Y - 1, Y2 >= 0.
move(right, X, Y, X, Y2) :- grid_size(S), Y2 is Y + 1, Y2 < S.

% Position validation
valid_pos(X, Y) :-
    \+ rock_pos(X, Y),
    grid_size(S),
    X >= 0, Y >= 0,
    X < S, Y < S.

% Main solving predicates
find_path(Actions) :-
    bird_pos(X, Y),
    findall((PX, PY), pig_pos(PX, PY), Pigs),
    find_optimal_path((X, Y), Pigs, [], Actions).

% Find the optimal path considering all pigs
find_optimal_path(_, [], Actions, Actions).
find_optimal_path((X, Y), [(PX, PY)|RemainingPigs], CurrentActions, FinalActions) :-
    ucs([[ (X, Y) ]], (PX, PY), [], Path),  % Find the shortest path to the current pig
    convert_path_to_actions(Path, PathActions),
    append(CurrentActions, PathActions, NewActions),
    find_optimal_path((PX, PY), RemainingPigs, NewActions, FinalActions).

% Uniform Cost Search implementation
ucs([[Start]], End, _, Path) :-
    ucs_search([0-[Start]], End, [], RevPath),
    reverse(RevPath, Path).

update_visited(Node, Cost, Visited, NewVisited) :-
    (   member(Node-OldCost, Visited), OldCost > Cost
    ->  select(Node-OldCost, Visited, TempVisited),
        NewVisited = [Node-Cost | TempVisited]
    ;   \+ member(Node-_, Visited)
    ->  NewVisited = [Node-Cost | Visited]
    ;   NewVisited = Visited
    ).

ucs_search([Cost-[End|Path]|_], End, _, [End|Path]).

ucs_search([Cost-[Current|Path]|Queue], Goal, Visited, FinalPath) :-
    Current = (X, Y),
    findall(
        NewCost-[Next, Current|Path],
        (move(_, X, Y, NX, NY),
         Next = (NX, NY),
         valid_pos(NX, NY),
         NewCost is Cost + 1,  % Cost of each move is 1
         \+ member_cost(Next, NewCost, Visited)),
        NewNodes
    ),
    append(NewNodes, Queue, UnsortedQueue),
    keysort(UnsortedQueue, SortedQueue),
    update_visited(Current, Cost, Visited, NewVisited),
    ucs_search(SortedQueue, Goal, NewVisited, FinalPath).

% Check if node exists in Visited with lower or equal cost
member_cost(Node, NewCost, Visited) :-
    member(Node-VisitedCost, Visited),
    VisitedCost < NewCost.

% Convert path to actions
convert_path_to_actions([], []).
convert_path_to_actions([_], []).
convert_path_to_actions([(X1,Y1), (X2,Y2)|Rest], [ActionNum|Actions]) :-
    get_direction_number((X1,Y1), (X2,Y2), ActionNum),
    convert_path_to_actions([(X2,Y2)|Rest], Actions).

% Predicate to get numeric direction
get_direction_number((X1,Y1), (X2,Y2), Action) :-
    (   X2 =:= X1 - 1, Y2 =:= Y1 -> Action = 0 ;    % up
        X2 =:= X1 + 1, Y2 =:= Y1 -> Action = 1 ;    % down
        X2 =:= X1, Y2 =:= Y1 - 1 -> Action = 2 ;    % left
        X2 =:= X1, Y2 =:= Y1 + 1 -> Action = 3 ).   % right