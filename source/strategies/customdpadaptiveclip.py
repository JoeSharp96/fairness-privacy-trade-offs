from flwr.serverapp.strategy import DifferentialPrivacyServerSideAdaptiveClipping, Result
import math
import io
import time
from logging import INFO
from typing import Callable, Iterable, Optional
from flwr.app import ArrayRecord, ConfigRecord, Message, MetricRecord, RecordDict, MessageType
from flwr.common import log
from flwr.serverapp import Grid
from flwr.serverapp.strategy.strategy_utils import log_strategy_start_info, sample_nodes
from source.utils.strategy import get_individual_metrics

class CustomDifferentialPrivacyAdaptiveClipping(DifferentialPrivacyServerSideAdaptiveClipping):

    def configure_train(
        self, server_round: int, arrays: ArrayRecord, config: ConfigRecord, grid: Grid, malicious_nodes: dict
    ) -> Iterable[Message]:
        """Configure the next round of training."""
        self.current_arrays = arrays
        return self.strategy.configure_train(server_round, arrays, config, grid, malicious_nodes)

    def is_malicious(self, grid: Grid, fraction_malicious: float) -> dict:
        node_ids = grid.get_node_ids()
        if fraction_malicious < 0.0 or fraction_malicious > 1.0:
            ValueError(f"Invalid fraction_malicious value: {fraction_malicious}. Value must be between 0.0 and 1.0.")
            fraction_malicious = 0.0
        num_malicious = int(len(node_ids) * fraction_malicious)
        num_friendly = len(node_ids) - num_malicious
        malicious_nodes = {}
        for node, flag in zip(node_ids, [True] * num_malicious + [False] * num_friendly):
            malicious_nodes[node] = flag
        return malicious_nodes

    def start(
        self,
        grid: Grid,
        initial_arrays: ArrayRecord,
        server,
        num_rounds: int = 3,
        timeout: float = 3600,
        train_config: ConfigRecord | None = None,
        evaluate_config: ConfigRecord | None = None,
        evaluate_fn: Callable[[int, ArrayRecord], MetricRecord | None] | None = None,
        fraction_malicious: float = 0.0
    ) -> Result:
        """Execute the federated learning strategy.

        Runs the complete federated learning workflow for the specified number of
        rounds, including training, evaluation, and optional centralized evaluation.

        Parameters
        ----------
        grid : Grid
            The Grid instance used to send/receive Messages from nodes executing a
            ClientApp.
        initial_arrays : ArrayRecord
            Initial model parameters (arrays) to be used for federated learning.
        num_rounds : int (default: 3)
            Number of federated learning rounds to execute.
        timeout : float (default: 3600)
            Timeout in seconds for waiting for node responses.
        train_config : ConfigRecord, optional
            Configuration to be sent to nodes during training rounds.
            If unset, an empty ConfigRecord will be used.
        evaluate_config : ConfigRecord, optional
            Configuration to be sent to nodes during evaluation rounds.
            If unset, an empty ConfigRecord will be used.
        evaluate_fn : Callable[[int, ArrayRecord], Optional[MetricRecord]], optional
            Optional function for centralized evaluation of the global model. Takes
            server round number and array record, returns a MetricRecord or None. If
            provided, will be called before the first round and after each round.
            Defaults to None.

        Returns
        -------
        Results
            Results containing final model arrays and also training metrics, evaluation
            metrics and global evaluation metrics (if provided) from all rounds.
        """
        log(INFO, "Starting %s strategy:", self.__class__.__name__)
        log_strategy_start_info(
            num_rounds, initial_arrays, train_config, evaluate_config
        )
        self.summary()
        log(INFO, "")

        # Initialize if None
        train_config = ConfigRecord() if train_config is None else train_config
        evaluate_config = ConfigRecord() if evaluate_config is None else evaluate_config
        result = Result()

        t_start = time.time()
        # Evaluate starting global parameters
        if evaluate_fn:
            res = evaluate_fn(0, initial_arrays,server)
            log(INFO, "Initial global evaluation results: %s", res)
            if res is not None:
                result.evaluate_metrics_serverapp[0] = res

        arrays = initial_arrays
        malicious_nodes = self.is_malicious(grid, fraction_malicious)
        for current_round in range(1, num_rounds + 1):
            log(INFO, "")
            log(INFO, "[ROUND %s/%s]", current_round, num_rounds)

            # -----------------------------------------------------------------
            # --- TRAINING (CLIENTAPP-SIDE) -----------------------------------
            # -----------------------------------------------------------------

            # Call strategy to configure training round
            # Send messages and wait for replies
            train_replies = grid.send_and_receive(
                messages=self.configure_train(
                    current_round,
                    arrays,
                    train_config,
                    grid,
                    malicious_nodes
                ),
                timeout=timeout,
            )

            # Aggregate train
            agg_arrays, agg_train_metrics = self.aggregate_train(
                current_round,
                train_replies,
            )

            # Log training metrics and append to history
            if agg_arrays is not None:
                result.arrays = agg_arrays
                arrays = agg_arrays
            if agg_train_metrics is not None:
                log(INFO, "\t└──> Aggregated MetricRecord: %s", agg_train_metrics)
                result.train_metrics_clientapp[current_round] = agg_train_metrics

            # -----------------------------------------------------------------
            # --- EVALUATION (CLIENTAPP-SIDE) ---------------------------------
            # -----------------------------------------------------------------

            # Call strategy to configure evaluation round
            # Send messages and wait for replies
            evaluate_replies = grid.send_and_receive(
                messages=self.configure_evaluate(
                    current_round,
                    arrays,
                    evaluate_config,
                    grid
                ),
                timeout=timeout,
            )

            # Aggregate evaluate
            agg_evaluate_metrics = self.aggregate_evaluate(
                current_round,
                evaluate_replies,
            )

            # Log training metrics and append to history
            if agg_evaluate_metrics is not None:
                log(INFO, "\t└──> Aggregated MetricRecord: %s", agg_evaluate_metrics)
                result.evaluate_metrics_clientapp[current_round] = agg_evaluate_metrics

            # -----------------------------------------------------------------
            # --- EVALUATION (SERVERAPP-SIDE) ---------------------------------
            # -----------------------------------------------------------------

            # Centralized evaluation
            if evaluate_fn:
                log(INFO, "Global evaluation")
                res = evaluate_fn(current_round, arrays,server)
                log(INFO, "\t└──> MetricRecord: %s", res)
                if res is not None:
                    result.evaluate_metrics_serverapp[current_round] = res

        individual_metrics = get_individual_metrics(evaluate_replies)

        log(INFO, "")
        log(INFO, "Strategy execution finished in %.2fs", time.time() - t_start)
        log(INFO, "")
        log(INFO, "Final results:")
        log(INFO, "")
        for line in io.StringIO(str(result)):
            log(INFO, "\t%s", line.strip("\n"))
        log(INFO, "")

        return result, individual_metrics
