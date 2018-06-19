#!/usr/bin/env bash
set -x

export DOCKER_REGISTRY='retrocontestviebtzaqpdksjflr.azurecr.io'
export docker_registry_username='xxx'
export docker_registry_password='xxx'

docker login $DOCKER_REGISTRY \
    --username $docker_registry_username \
    --password $docker_registry_password

#!/usr/bin/env bash
set -x

export dir=dmr-agent

rm -rf $dir
mkdir $dir

cp ../dockerfile/dmr.docker dmr-agent

cp -r ../A3gent/lawking $dir
cp -r ../A3gent/cpt $dir
cp -r ../A3gent/detect $dir
cp -r ../A3gent/detect_model $dir

cp ../A3gent/dmr_agent.py $dir/agent.py

export version='0511_npn_01350_110592000_totalstep3e6_lr_clip'

cd $dir
docker build -f dmr.docker -t $DOCKER_REGISTRY/dmr-agent:$version .

docker push $DOCKER_REGISTRY/dmr-agent:$version

