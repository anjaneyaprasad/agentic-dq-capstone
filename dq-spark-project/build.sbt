import sbtassembly.AssemblyPlugin.autoImport._

ThisBuild / scalaVersion := "2.12.18" // Spark 3.x → Scala 2.12

lazy val sparkVersion = "3.5.1" // good match for Deequ 2.0.7-spark-3.5

lazy val root = (project in file("."))
  .settings(
    name := "dq-spark-project",
    version := "0.1.0-SNAPSHOT",
    libraryDependencies ++= Seq(
      "org.apache.spark" %% "spark-core" % sparkVersion % "provided",
      "org.apache.spark" %% "spark-sql" % sparkVersion % "provided",
      "com.amazon.deequ" % "deequ" % "2.0.7-spark-3.5",
      "net.snowflake" % "spark-snowflake_2.12" % "2.16.0-spark_3.4",
      "net.snowflake" % "snowflake-jdbc" % "3.16.1",
      "org.slf4j" % "slf4j-api" % "1.7.36",
      "org.scalatest" %% "scalatest" % "3.2.18" % Test
    ),
    testFrameworks += new TestFramework("org.scalatest.tools.Framework")
  )

// skip tests during assembly
assembly / test := {}

assemblyMergeStrategy := {
  case PathList("META-INF", xs @ _*) => MergeStrategy.discard
  case "reference.conf"              => MergeStrategy.concat
  case _                             => MergeStrategy.first
}
