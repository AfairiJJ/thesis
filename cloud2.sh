#! /usr/bin/bash

source $HOME/.virtualenvs/thesis/bin/activate

modelversions=(523 522)

for i in "${modelversions[@]}"; do
  python3 $HOME/jj/thesis2/main.py --modelversion "$i" &
done
wait

modelversions=(519 520)

for i in "${modelversions[@]}"; do
  python3 $HOME/jj/thesis2/main.py --modelversion "$i" &
done
wait

modelversions=(521 505)

for i in "${modelversions[@]}"; do
  python3 $HOME/jj/thesis2/main.py --modelversion "$i" &
done
wait

modelversions=(514 509)

for i in "${modelversions[@]}"; do
  python3 $HOME/jj/thesis2/main.py --modelversion "$i" &
done
wait

modelversions=(501 503)

for i in "${modelversions[@]}"; do
  python3 $HOME/jj/thesis2/main.py --modelversion "$i" &
done
wait

modelversions=(510 511)

for i in "${modelversions[@]}"; do
  python3 $HOME/jj/thesis2/main.py --modelversion "$i" &
done
wait

modelversions=(504 506)

for i in "${modelversions[@]}"; do
  python3 $HOME/jj/thesis2/main.py --modelversion "$i" &
done
wait

modelversions=(507 508)

for i in "${modelversions[@]}"; do
  python3 $HOME/jj/thesis2/main.py --modelversion "$i" &
done
wait

modelversions=(512 513)

for i in "${modelversions[@]}"; do
  python3 $HOME/jj/thesis2/main.py --modelversion "$i" &
done
wait
