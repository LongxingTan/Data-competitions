a=`hostname`

printf "%s %s\n%s %s\n%s %s\n%s %s\n%s %s\n%s %s\n" \
127.0.0.1 localhost \
127.0.0.1 $a \
127.0.0.1 occlum-node \
::1  localhost \
::1  $a \
::1  occlum-node 
