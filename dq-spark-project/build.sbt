ThisBuild / scalaVersion := "2.12.18"
ThisBuild / organization := "com.anjaneya"
ThisBuild / version      := "0.1.0-SNAPSHOT"

lazy val root = (project in file("."))
  .settings(
    name := "dq-spark-deequ-project",

    libraryDependencies ++= Seq(
      // Spark core + SQL
      "org.apache.spark" %% "spark-core" % "3.5.1",
      "org.apache.spark" %% "spark-sql"  % "3.5.1",

      // Deequ for data quality checks
      "com.amazon.deequ" % "deequ" % "2.0.7-spark-3.5",
      "org.scalatest"   %% "scalatest"  % "3.2.18" % Test
      // (Optional but useful) Logging
      // "org.slf4j" % "slf4j-simple" % "2.0.9"
    ),

    // Needed when running Spark from sbt
    fork := true,
    javaOptions ++= Seq(
      "-Xms1G",
      "-Xmx4G"
    )
  )
