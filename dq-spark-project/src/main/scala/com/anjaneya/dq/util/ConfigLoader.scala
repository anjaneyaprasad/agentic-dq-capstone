package com.anjaneya.dq.util

import com.anjaneya.dq.config._
import org.yaml.snakeyaml.Yaml

import java.io.InputStream
import scala.jdk.CollectionConverters._

object ConfigLoader {

  // reading datasets.yaml to DatasetsConfig, which means all tables' configs
  def loadDatasetsConfig(path: String = "configs/datasets.yaml"): DatasetsConfig = {
    val is: InputStream = getClass.getClassLoader.getResourceAsStream(path)

    if (is == null) {
      throw new IllegalArgumentException(s"Config file not found on classpath: $path")
    }

    val yaml = new Yaml()

    // Load root as a Java Map<String, Object>
    val root: java.util.Map[String, Object] =
      yaml.load(is).asInstanceOf[java.util.Map[String, Object]]

    val rootMap = root.asScala // Map[String, Object]

    // Get "datasets" key explicitly, no getOrElse (avoid Nothing inference)
    val datasetsAny: Object = rootMap.getOrElse(
      "datasets",
      throw new IllegalArgumentException("Missing 'datasets' key in YAML")
    )

    // datasets: List<Map<String, Object>>
    val datasetsJava =
      datasetsAny.asInstanceOf[java.util.List[java.util.Map[String, Object]]]

    val datasetsScala: Seq[DatasetConfig] = datasetsJava.asScala.toSeq.map { dsMapJava =>
      val m = dsMapJava.asScala // Map[String, Object]

      def strList(key: String): Seq[String] = {
        m.get(key) match {
          case Some(list: java.util.List[_]) =>
            list.asScala.map(_.toString).toSeq
          case Some(single) =>
            Seq(single.toString)
          case None =>
            Seq.empty
        }
      }

      DatasetConfig(
        name            = m("name").toString,
        source          = m("source").toString,
        table_or_path   = m("table_or_path").toString,
        primary_keys    = strList("primary_keys"),
        partitions      = strList("partitions"),
        critical_columns = strList("critical_columns")
      )
    }

    DatasetsConfig(datasetsScala)
  }

  // To get a specific DatasetConfig by name from DatasetsConfig
  def getDatasetConfig(name: String, cfg: DatasetsConfig): DatasetConfig =
    cfg.datasets.find(_.name == name).getOrElse {
      throw new IllegalArgumentException(s"Dataset not found: $name")
    }
}