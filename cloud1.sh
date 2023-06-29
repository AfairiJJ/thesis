#! /usr/bin/bash

source $HOME/.virtualenvs/thesis/bin/activate

modelversions=(423 422)

for i in "${modelversions[@]}"; do
  python3 $HOME/jj/thesis1/main.py --modelversion "$i" &
done
wait

modelversions=(419 420)

for i in "${modelversions[@]}"; do
  python3 $HOME/jj/thesis1/main.py --modelversion "$i" &
done
wait

modelversions=(421 405)

for i in "${modelversions[@]}"; do
  python3 $HOME/jj/thesis1/main.py --modelversion "$i" &
done
wait

modelversions=(414 409)

for i in "${modelversions[@]}"; do
  python3 $HOME/jj/thesis1/main.py --modelversion "$i" &
done
wait

modelversions=(401 403)

for i in "${modelversions[@]}"; do
  python3 $HOME/jj/thesis1/main.py --modelversion "$i" &
done
wait

modelversions=(410 411)

for i in "${modelversions[@]}"; do
  python3 $HOME/jj/thesis1/main.py --modelversion "$i" &
done
wait

modelversions=(404 406)

for i in "${modelversions[@]}"; do
  python3 $HOME/jj/thesis1/main.py --modelversion "$i" &
done
wait

modelversions=(407 408)

for i in "${modelversions[@]}"; do
  python3 $HOME/jj/thesis1/main.py --modelversion "$i" &
done
wait

modelversions=(412 413)

for i in "${modelversions[@]}"; do
  python3 $HOME/jj/thesis1/main.py --modelversion "$i" &
done
wait
