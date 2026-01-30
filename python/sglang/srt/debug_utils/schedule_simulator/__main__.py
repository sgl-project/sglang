from sglang.srt.debug_utils.schedule_simulator.entrypoint import create_arg_parser, main

if __name__ == "__main__":
    parser = create_arg_parser()
    args = parser.parse_args()
    main(args)
