# sparkify

Udacity Data Science Nanodegree Capstone Project

In order to run the project the following needs to ne installed in the local machine.

Download the Spark version 2.4.5 pre-built for Apache Hadoop 2.7 and later from [here](http://spark.apache.org/downloads.html)

Unpack it in a directory on your machine.

Modify your *.bash_profile* and add the following:

```sh
export SPARK_HOME="/Users/<YOUR_USER_NAME>/spark-2.1.1-bin-hadoop2.7"
export PATH=$SPARK_HOME/bin:$PATH
```

```sh
$ source .bash_profile
```

```sh
$ pip insta;l findspark
```

```sh
brew tap adoptopenjdk/openjdk
brew cask install adoptopenjdk8
```

```sh
ls -la /Library/Java/JavaVirtualMachines
total 0
drwxr-xr-x  3 root  wheel   96 Apr 15 15:25 adoptopenjdk-8.jdk
drwxr-xr-x  3 root  wheel   96 Nov 21 14:05 jdk-13.0.1.jdk
```

```sh
export JAVA_HOME=/Library/Java/JavaVirtualMachines/adoptopenjdk-8.jdk/Contents/Home
export PATH=$JAVA_HOME/bin:$PATH
```

```sh
java -version
openjdk version "1.8.0_242"
OpenJDK Runtime Environment (AdoptOpenJDK)(build 1.8.0_242-b08)
OpenJDK 64-Bit Server VM (AdoptOpenJDK)(build 25.242-b08, mixed mode)
```

```sh
$ pip install xgboost
```

```sh
$ brew install libomp
```

```sh
$ conda install lightgbm
```

```sh
$ conda install pyarrow
```
